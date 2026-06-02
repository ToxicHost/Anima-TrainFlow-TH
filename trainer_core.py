"""
Studio Trainer — framework-free engine.

Extracted from the original single-file Gradio app. Contains NO UI framework
imports (no gradio, no fastapi): config builders, ONNX preprocessing, dataset
analysis, settings persistence, and the training subprocess orchestration split
into build_launch / tail_process / kill_process so the server layer owns the
live process.

Hard constraints (do not "improve"): bf16 only (never fp16), full bf16 DiT
(never fp8 for training), LLM adapter frozen, blocks_to_swap caps at 26, kohya
TOML key names are an external contract.
"""

import os
import re
import json
import glob
import math
import time
import queue
import shutil
import platform
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional, Tuple

import toml
import psutil
import numpy as np
import pandas as pd
import cv2
import onnxruntime as rt
from PIL import Image

# torch is used only for the hard CUDA requirement check. The engine refuses CPU.
import torch


# ==========================================
# CONSTANTS
# ==========================================
VERSION = "2.0.0"   # bump on each release; drives the updater's up-to-date check

ROOT = Path(__file__).resolve().parent
PORTABLE_PYTHON = ROOT / "python_embeded" / "python.exe"

TRAIN_BASE = ROOT / "training"
OUTPUT_BASE = TRAIN_BASE / "output"
SETTINGS_FILE = TRAIN_BASE / "settings.json"

TRAIN_DIR = TRAIN_BASE / "sd-scripts"
TRAIN_SCRIPT = TRAIN_DIR / "anima_train_network.py"          # LoRA trainer
TRAIN_SCRIPT_FULL = TRAIN_DIR / "anima_train.py"             # full fine-tune trainer

# Block-swap is capped at num_blocks - 2 (28 - 2 = 26) by anima_models.enable_block_swap().
MAX_BLOCKS_TO_SWAP = 26

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
TAGGER_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

MAX_LOG_LINES = 500
LOG_BLACKLIST = [
    "triton not found",
    "flop counting will not work",
    "Lib\\site-packages\\torch\\utils\\flop_counter.py",
]

for d in [TRAIN_BASE, OUTPUT_BASE]:
    d.mkdir(parents=True, exist_ok=True)


# ==========================================
# SETTINGS (keyed payload — replaces the positional auto_save_state contract)
# ==========================================
DEFAULT_SETTINGS = {
    "trigger_word": "",
    "dataset_path": "",
    "dit_path": str(ROOT / "models" / "anima" / "dit" / "anima-preview.safetensors"),
    "qwen_path": str(ROOT / "models" / "anima" / "text_encoder" / "qwen_3_06b_base.safetensors"),
    "vae_path": str(ROOT / "models" / "anima" / "vae" / "qwen_image_vae.safetensors"),
    "network_rank": 32,
    "learning_rate": "1.0",   # STRING — coupled to optimizer (Prodigy=1.0, AdamW=0.00005)
    "optimizer": "Prodigy",
    "training_steps": 2400,
    "save_steps": 300,
    "sample_steps": 300,
    "pos_prompt": "",
    "neg_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name",
    "width": 1024,
    "height": 1024,
    "sample_steps_gen": 30,
    "sample_cfg": 4.0,
    "sample_seed": 42,
    "train_seed": 42,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "blocks_to_swap": 0,
    "full_finetune": False,
    "side_min": 512,
    "side_max": 768,
    "tagger_gen_thresh": 0.35,
    "tagger_char_thresh": 0.85,
    "tagger_overwrite": False,
    "prune_tags": "",
}

# Fixed values baked into the training config, never exposed as user controls.
# (Trap 4 cleaned: weighting_scheme de-duplicated to its effective value
# "logit_normal"; the dead blocks_to_swap copy removed — the real value is the
# clamped function argument.)
HIDDEN_SETTINGS = {
    "lr_scheduler": "cosine",
    "mixed_precision": "bf16",          # bf16 ONLY — never fp16.
    "save_precision": "bf16",
    "gradient_checkpointing": True,
    "network_module": "networks.lora_anima",
    "network_train_unet_only": True,    # keeps the LLM adapter frozen on the LoRA path.
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "weighting_scheme": "logit_normal",
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs": True,
    "cache_text_encoder_outputs_to_disk": True,
    "sdpa": True,
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": True,
    "max_grad_norm": 1.0,
    "vae_batch_size": 1,
    "sigmoid_scale": 1.3,
}


def get_settings() -> Dict:
    """Defaults merged over by the persisted settings.json."""
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings


def save_settings(new_values: Dict) -> Dict:
    """Merge a keyed payload over the current settings and persist. Returns the
    merged result. Only known keys are kept (defends against junk payloads)."""
    current = get_settings()
    for k, v in (new_values or {}).items():
        if k in DEFAULT_SETTINGS:
            current[k] = v
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=4)
    return current


# ==========================================
# DATASET HELPERS
# ==========================================
def count_images(dataset_path: str) -> int:
    path = Path(dataset_path)
    if not path.exists():
        return 0
    return len([f for f in path.glob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS])


def analyze_dataset_resolution(dataset_path: str) -> Tuple[int, int]:
    path = Path(dataset_path)
    if not path.exists():
        return 512, 768

    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [f for f in path.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    if not image_files:
        return 512, 768

    max_area = 0
    max_side = 0
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                max_area = max(max_area, w * h)
                max_side = max(max_side, w, h)
        except Exception:
            continue

    base_res = int(math.ceil(math.sqrt(max_area) / 64.0) * 64)
    max_bucket = int(math.ceil(max_side / 64.0) * 64)
    return base_res, max_bucket


# ==========================================
# CONFIG BUILDERS (kohya TOML — external key contract, do not rename)
# ==========================================
def create_sample_prompts(project_name, trigger_word, pos_prompt, neg_prompt, width, height, steps_gen, cfg, seed, out_dir):
    prompt_path = out_dir / f"{project_name}_prompts.txt"
    trigger = trigger_word.strip()
    user_prompt = pos_prompt.strip().replace("\n", " ")

    if trigger and not user_prompt.startswith(trigger):
        actual_pos = f"{trigger}, {user_prompt}" if user_prompt else trigger
    else:
        actual_pos = user_prompt if user_prompt else trigger

    actual_neg = neg_prompt.strip().replace("\n", " ")
    prompt_str = f"{actual_pos} --n {actual_neg} --w {int(width)} --h {int(height)} --l {float(cfg)} --s {int(steps_gen)} --d {int(seed)}"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_str)
    return str(prompt_path)


def create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, out_dir, max_steps, batch_size, grad_acc):
    config_path = out_dir / f"{project_name}_dataset.toml"
    prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else None

    num_images = count_images(dataset_path)
    if num_images == 0:
        num_images = 1

    effective_batch = int(batch_size) * int(grad_acc)
    calculated_repeats = (int(max_steps) * effective_batch) / num_images
    final_repeats = max(1, math.ceil(calculated_repeats))

    dataset_config = {
        "general": {
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": max_bucket,
            "bucket_reso_steps": 64,
            "bucket_no_upscale": True,
        },
        "datasets": [{
            "resolution": base_res,
            "subsets": [{
                "image_dir": Path(dataset_path).resolve().as_posix(),
                "caption_extension": ".txt",
                "num_repeats": final_repeats,
                "caption_prefix": prefix,
                "keep_tokens": 1,
                "caption_dropout_rate": 0.05,
            }],
        }],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(dataset_config, f)
    return str(config_path)


def _optimizer_args(optimizer):
    if optimizer == "Prodigy":
        scheduler = "constant"
        opt_args = ["decouple=True", "weight_decay=0.01", "d_coef=1.0", "use_bias_correction=True", "safeguard_warmup=True", "betas=0.9,0.99"]
    else:
        scheduler = "cosine"
        opt_args = ["weight_decay=0.01"]
    return scheduler, opt_args


def create_training_toml(project_name, config_save_dir, actual_output_dir, rank, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, blocks_to_swap=0):
    config_path = config_save_dir / f"{project_name}_training.toml"
    network_rank = int(rank)
    network_alpha = network_rank
    scheduler, opt_args = _optimizer_args(optimizer)

    training_config = {
        "pretrained_model_name_or_path": Path(models["dit_path"]).resolve().as_posix(),
        "qwen3": Path(models["qwen_path"]).resolve().as_posix(),
        "vae": Path(models["vae_path"]).resolve().as_posix(),
        "network_module": HIDDEN_SETTINGS["network_module"],
        "network_dim": network_rank,
        "network_alpha": network_alpha,
        "network_train_unet_only": HIDDEN_SETTINGS["network_train_unet_only"],
        "gradient_checkpointing": HIDDEN_SETTINGS["gradient_checkpointing"],
        "max_grad_norm": 1.0,
        "learning_rate": float(lr),
        "optimizer_type": optimizer,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        "max_train_steps": int(max_steps),
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "save_every_n_steps": int(save_steps),
        "sample_every_n_steps": int(sample_steps),
        "sample_prompts": Path(prompt_path).resolve().as_posix(),
        "sample_sampler": "euler",
        "timestep_sampling": HIDDEN_SETTINGS["timestep_sampling"],
        "discrete_flow_shift": HIDDEN_SETTINGS["discrete_flow_shift"],
        "sigmoid_scale": HIDDEN_SETTINGS["sigmoid_scale"],
        "weighting_scheme": HIDDEN_SETTINGS["weighting_scheme"],
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "attn_mode": "sdpa",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "max_data_loader_n_workers": 4,
        "vae_chunk_size": 32,
        "vae_disable_cache": True,
        "seed": int(train_seed),
    }
    # Only write blocks_to_swap when enabled, so the default config stays byte-identical to upstream.
    if int(blocks_to_swap) > 0:
        training_config["blocks_to_swap"] = int(blocks_to_swap)
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(training_config, f)
    return str(config_path)


def create_full_finetune_toml(project_name, config_save_dir, actual_output_dir, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, blocks_to_swap=0):
    """Config for the full DiT fine-tune trainer (anima_train.py): the same shared
    train_util/anima keys MINUS every network_* key, with the LLM adapter frozen
    (llm_adapter_lr=0.0)."""
    config_path = config_save_dir / f"{project_name}_training.toml"
    scheduler, opt_args = _optimizer_args(optimizer)

    training_config = {
        "pretrained_model_name_or_path": Path(models["dit_path"]).resolve().as_posix(),
        "qwen3": Path(models["qwen_path"]).resolve().as_posix(),
        "vae": Path(models["vae_path"]).resolve().as_posix(),
        "llm_adapter_lr": 0.0,  # HARD CONSTRAINT: keep the LLM adapter frozen.
        "gradient_checkpointing": HIDDEN_SETTINGS["gradient_checkpointing"],
        "max_grad_norm": 1.0,
        "learning_rate": float(lr),
        "optimizer_type": optimizer,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        "max_train_steps": int(max_steps),
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],  # bf16 only — never fp16.
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "save_every_n_steps": int(save_steps),
        "sample_every_n_steps": int(sample_steps),
        "sample_prompts": Path(prompt_path).resolve().as_posix(),
        "sample_sampler": "euler",
        "timestep_sampling": HIDDEN_SETTINGS["timestep_sampling"],
        "discrete_flow_shift": HIDDEN_SETTINGS["discrete_flow_shift"],
        "sigmoid_scale": HIDDEN_SETTINGS["sigmoid_scale"],
        "weighting_scheme": HIDDEN_SETTINGS["weighting_scheme"],
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "attn_mode": "sdpa",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "max_data_loader_n_workers": 4,
        "vae_chunk_size": 32,
        "vae_disable_cache": True,
        "seed": int(train_seed),
    }
    if int(blocks_to_swap) > 0:
        training_config["blocks_to_swap"] = int(blocks_to_swap)
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(training_config, f)
    return str(config_path)


def project_name_from_trigger(trigger_word: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '_', (trigger_word or "").strip()).strip('_') or "untitled"


def get_latest_images(sample_dir: Path) -> List[str]:
    """Absolute paths of preview images, newest first. The server maps these to
    /preview?path=… with a containment guard."""
    if not sample_dir.exists():
        return []
    images = (glob.glob(str(sample_dir / "*.png"))
              + glob.glob(str(sample_dir / "*.jpg"))
              + glob.glob(str(sample_dir / "*.webp")))
    images.sort(key=os.path.getmtime, reverse=True)
    return images


# ==========================================
# CROPPER (U2Net) — ONNX preprocessing is tuned to the model file; do not refactor.
# ==========================================
class SmartCropper:
    def __init__(self) -> None:
        self.session: Optional[rt.InferenceSession] = None
        self.model_path: Path = ROOT / "models" / "u2net" / "u2net.onnx"

    def load_model(self) -> str:
        if self.session is not None:
            return "Already loaded"
        providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        self.session = rt.InferenceSession(str(self.model_path), providers=providers)
        return "GPU" if "CUDA" in self.session.get_providers()[0] else "CPU"

    def get_valid_buckets(self, side_min: int, side_max: int) -> List[Tuple[int, int]]:
        s_min, s_max = int(side_min), int(side_max)
        buckets = {(s_min, s_min)}
        for s in range(s_min + 64, s_max + 64, 64):
            buckets.add((s_min, s))
            buckets.add((s, s_min))
        return sorted(list(buckets), key=lambda x: (x[0] * x[1]))

    def get_best_bucket(self, w: int, h: int, buckets: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], str]:
        orig_ar = w / h
        log_orig = math.log(orig_ar)
        best_b = buckets[0]
        min_diff = float('inf')
        for b in buckets:
            b_ar = b[0] / b[1]
            diff = abs(math.log(b_ar) - log_orig)
            if diff < min_diff:
                min_diff = diff
                best_b = b
        reason = f"AR: {orig_ar:.2f} -> {best_b[0]/best_b[1]:.2f}"
        return best_b, reason

    def process_image(self, original_img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h_orig, w_orig = original_img.shape[:2]
        if abs((w_orig / h_orig) - (tw / th)) < 0.01:
            return cv2.resize(original_img, (tw, th), interpolation=cv2.INTER_AREA)

        low_res_scale = 1024 / max(h_orig, w_orig)
        img_sm = cv2.resize(original_img, (int(w_orig*low_res_scale), int(h_orig*low_res_scale)), interpolation=cv2.INTER_AREA)

        input_size = 320
        img_inp = cv2.resize(img_sm, (input_size, input_size), interpolation=cv2.INTER_AREA)
        img_inp = img_inp.astype(np.float32) / 255.0
        img_inp -= [0.485, 0.456, 0.406]
        img_inp /= [0.229, 0.224, 0.225]
        input_tensor = np.expand_dims(np.transpose(img_inp, (2, 0, 1)), 0)

        mask = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0][0][0]
        mask = cv2.resize(mask, (img_sm.shape[1], img_sm.shape[0]))

        y_idx, x_idx = np.where(mask > 0.15)
        if len(y_idx) > 0:
            top_y = int(np.min(y_idx) / low_res_scale)
            center_x = int(np.mean(x_idx) / low_res_scale)
        else:
            top_y, center_x = h_orig // 4, w_orig // 2

        scale = max(tw / w_orig, th / h_orig)
        cw, ch = int(tw / scale), int(th / scale)
        y1 = max(0, min(top_y - int(ch * 0.05), h_orig - ch))
        x1 = max(0, min(center_x - cw // 2, w_orig - cw))
        return cv2.resize(original_img[y1:y1+ch, x1:x1+cw], (tw, th), interpolation=cv2.INTER_AREA)

    def unload_model(self) -> None:
        self.session = None


smart_cropper = SmartCropper()


def generate_bucket_summary(bucket_counts: dict) -> List[str]:
    if not bucket_counts:
        return []
    lines = ["", "Bucket distribution summary:"]
    total = 0
    for bkt in sorted(bucket_counts.keys(), key=lambda x: x[0] * x[1]):
        count = bucket_counts[bkt]
        total += count
        lines.append(f"  - {bkt[0]}x{bkt[1]}: {count} images")
    lines.append(f"  = Total dataset size: {total} images")
    lines.append("")
    return lines


# ==========================================
# TAGGER (WD-EVA02) — ONNX preprocessing is tuned to the model file; do not refactor.
# ==========================================
class WDTagger:
    def __init__(self):
        self.model = None
        self.tag_names = []
        self.general_indexes = []
        self.character_indexes = []
        self.target_size = 448
        self.model_dir = ROOT / "models" / "wd-eva02-large-tagger-v3"
        self.model_path = self.model_dir / "model.onnx"
        self.csv_path = self.model_dir / "selected_tags.csv"
        self.kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]

    def load_model(self):
        if self.model is not None:
            return "Already loaded"
        if not self.model_path.exists() or not self.csv_path.exists():
            raise FileNotFoundError(f"Model or CSV not found in {self.model_dir}.")

        df = pd.read_csv(self.csv_path)
        name_series = df["name"].map(lambda x: str(x).replace("_", " ") if pd.notna(x) and str(x) not in self.kaomojis else str(x))
        self.tag_names = name_series.tolist()
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        self.model = rt.InferenceSession(str(self.model_path), providers=providers)
        current_provider = self.model.get_providers()[0]
        return "GPU (CUDA)" if "CUDA" in current_provider else "CPU (Slow Mode)"

    def preprocess(self, image):
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        image = canvas.convert("RGB")
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        if max_dim != self.target_size:
            padded_image = padded_image.resize((self.target_size, self.target_size), Image.BICUBIC)
        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # BGR
        return np.expand_dims(image_array, axis=0)

    def predict(self, image, gen_thresh, char_thresh):
        image_array = self.preprocess(image)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: image_array})
        preds = outputs[0][0]

        general_tags = [self.tag_names[i] for i in self.general_indexes if i < len(preds) and preds[i] > gen_thresh]
        char_tags = [self.tag_names[i] for i in self.character_indexes if i < len(preds) and preds[i] > char_thresh]

        char_tags = [char.replace("(", r"\(").replace(")", r"\)") for char in char_tags]
        general_tags = sorted(general_tags, key=lambda x: preds[self.tag_names.index(x)], reverse=True)

        final_list = []
        if char_tags:
            final_list.append(", ".join(char_tags))
        if general_tags:
            final_list.append(", ".join(general_tags).replace("(", r"\(").replace(")", r"\)"))
        return ", ".join(final_list)

    def unload_model(self):
        import gc
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


wd_tagger = WDTagger()


# ==========================================
# STRUCTURED EVENT HELPERS
# Event shape: {type: log|progress|preview|done|error, line, images, message, exit_code, tail}
# ==========================================
def ev_log(line: str) -> Dict:
    return {"type": "log", "line": line}


def ev_progress(line: str) -> Dict:
    return {"type": "progress", "line": line}


def ev_preview(images: List[str]) -> Dict:
    return {"type": "preview", "images": images}


def ev_done(message: str = "complete") -> Dict:
    return {"type": "done", "message": message}


def ev_error(message: str, exit_code: Optional[int] = None, tail: Optional[List[str]] = None) -> Dict:
    out = {"type": "error", "message": message}
    if exit_code is not None:
        out["exit_code"] = exit_code
    if tail is not None:
        out["tail"] = tail
    return out


# ==========================================
# DATASET TOOLS (event generators — emit new lines only, caller accumulates)
# ==========================================
def run_smart_crop(dataset_dir: str, side_min: float, side_max: float) -> Iterator[Dict]:
    path = Path(dataset_dir)
    if not dataset_dir or not path.exists():
        yield ev_error("Dataset path invalid.")
        return

    backup_dir = path / "original_images"
    backup_dir.mkdir(parents=True, exist_ok=True)

    available_buckets = smart_cropper.get_valid_buckets(int(side_min), int(side_max))
    yield ev_log(f"Buckets configured: {len(available_buckets)}")

    # 1. Move new original images out of the working dir.
    for f in path.glob('*'):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS and f.parent != backup_dir:
            try:
                if f.suffix.lower() == '.png':
                    with Image.open(f) as img:
                        if img.size in available_buckets:
                            continue
            except Exception:
                pass
            shutil.move(str(f), str(backup_dir / f.name))

    # 2. Prepare tasks (idempotent) + collect stats.
    originals = [f for f in backup_dir.glob('*') if f.suffix.lower() in IMAGE_EXTS]
    tasks = []
    bucket_counts: Dict[Tuple[int, int], int] = {}
    skipped = 0
    for orig in originals:
        try:
            with Image.open(orig) as img:
                w, h = img.size
            (tw, th), _ = smart_cropper.get_best_bucket(w, h, available_buckets)
            out_p = path / f"{orig.stem}.png"
            if out_p.exists():
                with Image.open(out_p) as check:
                    if check.size == (tw, th):
                        skipped += 1
                        bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                        continue
            tasks.append((orig, out_p, tw, th))
        except Exception:
            continue

    if not tasks:
        yield ev_log(f"All {skipped} images are already bucketed correctly.")
        for line in generate_bucket_summary(bucket_counts):
            yield ev_log(line)
        yield ev_done("bucketing complete")
        return

    # 3. Multi-threaded inference.
    try:
        yield ev_log(f"Model: {smart_cropper.load_model()}")
        yield ev_log(f"Processing {len(tasks)} images (skipped: {skipped})...")

        processed = 0
        lock = threading.Lock()

        def worker(task_data):
            in_p, out_p, tw, th = task_data
            nonlocal processed
            try:
                img = cv2.imread(str(in_p))
                if img is not None:
                    res = smart_cropper.process_image(img, tw, th)
                    cv2.imwrite(str(out_p), res, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                with lock:
                    processed += 1
                    bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                return None
            except Exception as e:
                return f"{in_p.name}: {str(e)}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, t) for t in tasks]
            for future in as_completed(futures):
                err = future.result()
                if err:
                    yield ev_log(f"warning: {err}")
                if processed % 5 == 0 or processed == len(tasks):
                    yield ev_progress(f"Progress: {processed}/{len(tasks)}")

        yield ev_log(f"Done. Processed: {processed}")
    except Exception as e:
        yield ev_error(f"Crop error: {str(e)}")
        return
    finally:
        smart_cropper.unload_model()

    for line in generate_bucket_summary(bucket_counts):
        yield ev_log(line)
    yield ev_done("bucketing complete")


def run_auto_tagging(dataset_dir, gen_thresh, char_thresh, overwrite) -> Iterator[Dict]:
    if not dataset_dir or not os.path.exists(dataset_dir):
        yield ev_error("Dataset path invalid.")
        return

    all_files = [f for f in Path(dataset_dir).glob('*') if f.is_file() and f.suffix.lower() in TAGGER_EXTS]
    image_files = []
    skipped = 0
    for f in all_files:
        if f.with_suffix('.txt').exists() and not overwrite:
            skipped += 1
        else:
            image_files.append(f)

    if not image_files and skipped == 0:
        yield ev_error("No images found.")
        return
    if not image_files and skipped > 0:
        yield ev_log(f"All images already have captions. Skipped: {skipped}")
        yield ev_done("nothing to tag")
        return

    yield ev_log("Initializing WD-Tagger (multi-threaded)...")
    try:
        mode = wd_tagger.load_model()
        yield ev_log(f"Model loaded using: {mode}")
        yield ev_log(f"Processing {len(image_files)} images in 4 threads (skipped: {skipped})...")

        processed_count = 0
        total = len(image_files)
        lock = threading.Lock()

        def process_single_image(img_path):
            nonlocal processed_count
            try:
                with Image.open(img_path) as img:
                    tags = wd_tagger.predict(img, gen_thresh, char_thresh)
                with open(img_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(tags)
                with lock:
                    processed_count += 1
                return None
            except Exception as e:
                return f"{img_path.name}: {str(e)}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_image, p) for p in image_files]
            for future in as_completed(futures):
                err = future.result()
                if err:
                    yield ev_log(f"warning: {err}")
                if processed_count % 5 == 0 or processed_count == total:
                    yield ev_progress(f"Tag progress: {processed_count}/{total} images.")

        yield ev_log(f"Tagging complete. Processed: {processed_count} | Skipped: {skipped}")
    except Exception as e:
        yield ev_error(f"Tagger error: {str(e)}")
        return
    finally:
        wd_tagger.unload_model()

    yield ev_done("captioning complete")


def run_prune_tags(dataset_dir, prune_tags) -> Iterator[Dict]:
    path = Path(dataset_dir)
    if not dataset_dir or not path.exists():
        yield ev_error("Dataset path invalid.")
        return

    tags = [t.strip() for t in (prune_tags or "").split(',') if t.strip()]
    if not tags:
        yield ev_log("No tags specified, nothing to do.")
        yield ev_done("nothing to prune")
        return

    txt_files = [f for f in path.glob('*.txt')]
    if not txt_files:
        yield ev_error("No .txt captions found in dataset path.")
        return

    # Back up originals first (only once) so the operation is reversible.
    backup_dir = path / "captions_backup"
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in txt_files:
            try:
                shutil.copy2(str(f), str(backup_dir / f.name))
            except Exception as e:
                yield ev_log(f"warning: backup failed for {f.name}: {e}")
        yield ev_log(f"Backed up {len(txt_files)} captions to captions_backup/")
    else:
        yield ev_log("captions_backup/ already exists — keeping the existing backup.")
    yield ev_log(f"Pruning {len(tags)} tag(s) from {len(txt_files)} captions...")

    patterns = [re.compile(r'\b' + re.escape(tag) + r'\b', re.IGNORECASE) for tag in tags]
    files_changed = 0
    tags_removed = 0
    for f in txt_files:
        try:
            original = f.read_text(encoding='utf-8')
        except Exception as e:
            yield ev_log(f"warning: {f.name}: {e}")
            continue

        text = original
        for pat in patterns:
            text, n = pat.subn('', text)
            tags_removed += n

        # Normalize: split on commas, drop empties, rejoin (no ", ," artifacts).
        parts = [p.strip() for p in text.split(',')]
        new_text = ', '.join([p for p in parts if p])
        if new_text != original.strip():
            try:
                f.write_text(new_text, encoding='utf-8')
                files_changed += 1
            except Exception as e:
                yield ev_log(f"warning: {f.name}: {e}")

    yield ev_log(f"Prune complete. Files changed: {files_changed} | Tag occurrences removed: {tags_removed}")
    yield ev_done("pruning complete")


# ==========================================
# FOLDER OPEN (local-only)
# ==========================================
def open_folder(target_dir: Path) -> None:
    if platform.system() == "Windows":
        os.startfile(str(target_dir))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(target_dir)])


def open_output_folder(trigger_word: str) -> str:
    proj = project_name_from_trigger(trigger_word)
    target_dir = OUTPUT_BASE / proj if proj else OUTPUT_BASE
    if not target_dir.exists():
        target_dir = OUTPUT_BASE
    open_folder(target_dir)
    return f"Opened {target_dir}"


def open_dataset_folder(dataset_dir: str) -> str:
    if not dataset_dir or not os.path.exists(dataset_dir):
        return "Dataset path invalid or empty."
    open_folder(Path(dataset_dir))
    return "Opened dataset folder."


# ==========================================
# TRAINING ORCHESTRATION — engine is stateless about the process.
# build_launch -> (cmd, env, cwd, meta); server owns the Popen via TrainingManager.
# ==========================================
def cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def validate_training(settings: Dict) -> List[str]:
    """Return a list of human-readable blocking errors (empty == OK)."""
    errors: List[str] = []
    dit_p = settings.get("dit_path")
    qwen_p = settings.get("qwen_path")
    vae_p = settings.get("vae_path")
    if not dit_p or not os.path.isfile(dit_p):
        errors.append(f"DiT file not found: {dit_p}")
    if not qwen_p or not os.path.isfile(qwen_p):
        errors.append(f"Qwen3 file not found: {qwen_p}")
    if not vae_p or not os.path.isfile(vae_p):
        errors.append(f"VAE file not found: {vae_p}")

    dataset_path = settings.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        errors.append(f"Dataset path not found: {dataset_path}")
    else:
        image_files = [f for f in Path(dataset_path).glob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not image_files:
            errors.append("No valid images found in the dataset path.")
        else:
            missing = [img.name for img in image_files if not img.with_suffix('.txt').exists()]
            if missing:
                errors.append(f"Found {len(missing)} images without .txt captions. Use Auto-Caption first.")
            oversized = []
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        w_img, h_img = img.size
                        if w_img >= 2048 or h_img >= 2048:
                            oversized.append(img.name)
                except Exception:
                    pass
            if oversized:
                errors.append(f"Found {len(oversized)} images >= 2048px. Run Smart Aspect Ratio Bucketing first.")

    if not cuda_available():
        errors.append("NVIDIA GPU / CUDA not detected. CPU training is not supported.")
    return errors


def clamp_blocks_to_swap(value) -> Tuple[int, Optional[str]]:
    try:
        bts = int(value or 0)
    except (TypeError, ValueError):
        bts = 0
    if bts < 0:
        return 0, None
    if bts > MAX_BLOCKS_TO_SWAP:
        return MAX_BLOCKS_TO_SWAP, f"Blocks to Swap clamped from {bts} to {MAX_BLOCKS_TO_SWAP} (max = num_blocks - 2)."
    return bts, None


def build_launch(settings: Dict) -> Dict:
    """Write the TOML configs and assemble the subprocess launch. Returns:
    {cmd, env, cwd, project_name, sample_dir, output_dir, warnings}."""
    warnings: List[str] = []

    trigger_word = settings.get("trigger_word", "")
    dataset_path = settings.get("dataset_path", "")
    project_name = project_name_from_trigger(trigger_word)

    project_out_dir = OUTPUT_BASE / project_name
    sample_dir = project_out_dir / "sample"
    project_configs_dir = project_out_dir / "configs"
    for d in [project_out_dir, sample_dir, project_configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_res, max_bucket = analyze_dataset_resolution(dataset_path)
    warnings.append(f"Auto-resolution: base {base_res}px, max bucket {max_bucket}px.")

    bts, clamp_note = clamp_blocks_to_swap(settings.get("blocks_to_swap", 0))
    if clamp_note:
        warnings.append(clamp_note)
    if bts > 0:
        warnings.append(f"Block swap enabled: blocks_to_swap={bts} (lower VRAM, slower steps).")

    models = {
        "dit_path": settings.get("dit_path"),
        "qwen_path": settings.get("qwen_path"),
        "vae_path": settings.get("vae_path"),
    }
    lr = settings.get("learning_rate", "1.0")
    optimizer = settings.get("optimizer", "Prodigy")
    t_steps = settings.get("training_steps", 2400)
    save_steps = settings.get("save_steps", 300)
    sample_steps = settings.get("sample_steps", 300)
    train_seed = settings.get("train_seed", 42)
    batch_size = settings.get("train_batch_size", 1)
    grad_acc = settings.get("gradient_accumulation_steps", 1)
    full_finetune = bool(settings.get("full_finetune", False))

    prompt_path = create_sample_prompts(
        project_name, trigger_word,
        settings.get("pos_prompt", ""), settings.get("neg_prompt", ""),
        settings.get("width", 1024), settings.get("height", 1024),
        settings.get("sample_steps_gen", 30), settings.get("sample_cfg", 4.0),
        settings.get("sample_seed", 42), project_configs_dir,
    )
    dataset_toml = create_dataset_toml(
        project_name, dataset_path, trigger_word, base_res, max_bucket,
        project_configs_dir, t_steps, batch_size, grad_acc,
    )

    if full_finetune:
        warnings.append("Full Fine-tune mode: training the full DiT (no LoRA). Needs far more VRAM (~31GB @512px); use Blocks to Swap on smaller cards.")
        training_toml = create_full_finetune_toml(
            project_name, project_configs_dir, project_out_dir, lr, optimizer,
            t_steps, save_steps, sample_steps, models, prompt_path, train_seed,
            batch_size, grad_acc, bts,
        )
        train_script = TRAIN_SCRIPT_FULL
    else:
        training_toml = create_training_toml(
            project_name, project_configs_dir, project_out_dir,
            settings.get("network_rank", 32), lr, optimizer,
            t_steps, save_steps, sample_steps, models, prompt_path, train_seed,
            batch_size, grad_acc, bts,
        )
        train_script = TRAIN_SCRIPT

    cmd = [
        str(PORTABLE_PYTHON.resolve()), "-m", "accelerate.commands.launch",
        "--num_processes=1", "--mixed_precision=bf16", "--dynamo_backend=no",
        train_script.resolve().as_posix(),
        "--config_file", Path(training_toml).resolve().as_posix(),
        "--dataset_config", Path(dataset_toml).resolve().as_posix(),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRAIN_DIR.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    env["KMP_WARNINGS"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["ACCELERATE_USE_CPU"] = "False"

    return {
        "cmd": cmd,
        "env": env,
        "cwd": str(TRAIN_DIR.resolve()),
        "project_name": project_name,
        "sample_dir": sample_dir,
        "output_dir": project_out_dir,
        "warnings": warnings,
    }


def spawn(launch: Dict) -> subprocess.Popen:
    return subprocess.Popen(
        launch["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1, cwd=launch["cwd"], env=launch["env"],
        encoding="utf-8", errors="ignore",
    )


def tail_process(proc: subprocess.Popen, sample_dir: Path, tail: Optional[List[str]] = None,
                 poll_interval: float = 1.0) -> Iterator[Dict]:
    """Stream {type:log} and {type:preview} events. stdout is read on a background
    thread so we can poll the sample dir on a timer (Trap 5 — previews are no longer
    scraped from log text). Terminal done/error is decided by the server, which owns
    stop_requested. Appends log lines to `tail` (capped) for the server's error tail."""
    if tail is None:
        tail = []

    seen_images = set(get_latest_images(sample_dir))
    if seen_images:
        yield ev_preview(sorted(seen_images, key=os.path.getmtime, reverse=True))

    line_q: "queue.Queue" = queue.Queue()
    SENTINEL = object()

    def reader():
        try:
            for line in iter(proc.stdout.readline, ""):
                line_q.put(line)
        finally:
            line_q.put(SENTINEL)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    step_pattern = re.compile(r"(\d+)/(\d+)")
    done = False
    while not done:
        try:
            item = line_q.get(timeout=poll_interval)
        except queue.Empty:
            # No new log line — poll the sample dir for new previews.
            current = get_latest_images(sample_dir)
            if set(current) != seen_images:
                seen_images = set(current)
                yield ev_preview(current)
            continue

        if item is SENTINEL:
            done = True
            break

        line_str = item.replace('\r', '').strip()
        if not line_str:
            continue
        if any(skip in line_str for skip in LOG_BLACKLIST):
            continue

        tail.append(line_str)
        if len(tail) > MAX_LOG_LINES:
            del tail[:-MAX_LOG_LINES]

        if "steps:" in line_str and step_pattern.search(line_str):
            yield ev_progress(line_str)
        else:
            yield ev_log(line_str)

    # Final preview sweep after the stream drains.
    current = get_latest_images(sample_dir)
    if set(current) != seen_images:
        yield ev_preview(current)


def kill_process(proc: subprocess.Popen) -> str:
    """psutil tree-kill of the training process and its children."""
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=3)
        for survivor in alive:
            try:
                survivor.kill()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        parent.wait(timeout=3)
        return "Stopping training... (clearing VRAM)"
    except psutil.NoSuchProcess:
        return "Process already finished."
    except Exception as e:
        return f"Error during stop: {str(e)}"


# ==========================================
# LAYERED PRESET SYSTEM (model C: type = intent, hardware = reality cap)
# Axis 1 (LoRA Type) sets the RECIPE; axis 2 (Hardware/VRAM) sets the CONSTRAINT.
# Presets only write values into existing fields — they never lock controls.
# ==========================================
LORA_TYPE_PRESETS = {
    # rank, optimizer, lr (STRING — coupled to optimizer), save_every, passes_per_image, note
    "Style": dict(
        rank=32, optimizer="Prodigy", lr="1.0", save=300, passes=130,
        note="Style: vary subject, keep the look constant. Trigger optional. ~1.8-2.4k steps."),
    "Character": dict(
        rank=32, optimizer="AdamW8bit", lr="0.00002", save=250, passes=120,
        note="Character: prune constant identity tags so the trigger absorbs them; keep clothing/pose/bg tagged."),
    "Concept (switchable)": dict(
        rank=32, optimizer="AdamW8bit", lr="0.00002", save=400, passes=140,
        note="Concept: keep the concept tag present; vary everything else. Trigger on."),
    "Concept (dominant)": dict(
        rank=64, optimizer="AdamW8bit", lr="0.00002", save=500, passes=140,
        note="Dominant concept: triggerless, merge in heavy (~0.9-1.0). Needs a larger dataset (150+)."),
}

# Hardware constraint layer: sets/overrides hardware knobs + a rank_cap that limits
# the type's rank. Never touches optimizer/LR/trigger/save.
VRAM_PRESETS = {
    # blocks_to_swap, batch, preview px, side_min, side_max, rank_cap
    "6 GB":  dict(swap=20, batch=1, prev=512,  smin=512, smax=512,  rank_cap=16),
    "12 GB": dict(swap=8,  batch=1, prev=768,  smin=512, smax=768,  rank_cap=32),
    "16 GB": dict(swap=0,  batch=1, prev=1024, smin=768, smax=1024, rank_cap=64),
    "24 GB": dict(swap=0,  batch=2, prev=1024, smin=768, smax=1024, rank_cap=128),
}

PRESET_CAVEATS = [
    "Presets are starting points — tune block-swap if you OOM or it's slow; adjust steps if needed.",
    "Bucket targets only take effect after re-running Smart Aspect Ratio Bucketing.",
    "Full fine-tune is independent — if it's on, the LoRA rank is irrelevant; raise block-swap manually for full-FT VRAM.",
]


def compute_steps(passes_per_image: int, img_count: int, batch: int, grad_acc: int) -> int:
    effective_batch = max(1, int(batch) * int(grad_acc))
    raw = passes_per_image * img_count
    return max(64, round(raw / effective_batch / 64) * 64)


def resolve_presets(lora_type: str, vram_tier: str, dataset_path: str, batch, grad_acc) -> Dict:
    """Resolve the two-axis layered presets into concrete field updates.

    Recipe knobs (optimizer/lr/save/passes->steps) come from the LoRA-Type preset;
    hardware knobs (blocks_to_swap/batch/preview/buckets) from the VRAM preset;
    rank = min(type_rank, hardware_rank_cap). Either axis being 'Custom' contributes
    nothing. Returns {updates, info, notes, caveats}."""
    updates: Dict = {}
    info: List[str] = []
    notes: List[str] = []

    t = LORA_TYPE_PRESETS.get(lora_type)
    v = VRAM_PRESETS.get(vram_tier)

    # Recipe knobs from the type preset only.
    type_rank = None
    if t:
        updates["optimizer"] = t["optimizer"]
        updates["learning_rate"] = t["lr"]
        updates["save_steps"] = t["save"]
        type_rank = t["rank"]
        info.append(t["note"])

    # Hardware knobs from the VRAM preset only.
    cap = None
    eff_batch = batch
    if v:
        updates["blocks_to_swap"] = v["swap"]
        updates["train_batch_size"] = v["batch"]
        updates["width"] = v["prev"]
        updates["height"] = v["prev"]
        updates["side_min"] = v["smin"]
        updates["side_max"] = v["smax"]
        cap = v["rank_cap"]
        eff_batch = v["batch"]   # the override drives the steps math below

    # Rank is the one shared knob: min(type_rank, cap).
    if type_rank is not None:
        if cap is not None:
            final_rank = min(type_rank, cap)
            updates["network_rank"] = final_rank
            if final_rank < type_rank:
                notes.append(f"Rank capped to {cap} for {vram_tier} — {lora_type} normally prefers {type_rank}.")
        else:
            updates["network_rank"] = type_rank

    # Steps from image count (only when a type is selected — passes come from type).
    if t:
        img_count = count_images(dataset_path)
        if img_count > 0:
            try:
                b = int(eff_batch if eff_batch is not None else 1)
            except (TypeError, ValueError):
                b = 1
            try:
                g = int(grad_acc or 1)
            except (TypeError, ValueError):
                g = 1
            updates["training_steps"] = compute_steps(t["passes"], img_count, b, g)
        else:
            notes.append("Set a dataset path to auto-compute steps.")

    return {"updates": updates, "info": info, "notes": notes, "caveats": PRESET_CAVEATS}
