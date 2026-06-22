"""
WD-EVA02 Booru auto-tagger — standalone, gallery-oriented.

A drop-in tagging engine extracted from the Anima-TrainFlow studio and reshaped
for an image-gallery / search use case (e.g. TrackImage). Unlike the trainer
version, this module:

  * returns STRUCTURED results (tag, category, score) instead of a single
    LoRA-prompt caption string,
  * keeps Booru underscores by default (`long_hair`, not "long hair"),
  * exposes the rating category (general / sensitive / questionable / explicit),
  * has NO torch and NO pandas dependency (stdlib csv + numpy only),
  * ships an optional tiny SQLite store for tag search / autocomplete.

Model
-----
This runs SmilingWolf's **wd-eva02-large-tagger-v3**. The weights are NOT
included. Download two files into the model directory:

    model.onnx            (~1.2 GB)
    selected_tags.csv     (the tag vocabulary)

from https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3

Dependencies
------------
    pip install onnxruntime        # CPU
    # or
    pip install onnxruntime-gpu    # NVIDIA GPU (CUDA)
    pip install numpy pillow       # TrackImage already ships these

The ONNX preprocessing (square-pad, 448px, BGR) is tuned to this exact model.
Do not "improve" it.

Model credit: SmilingWolf. See the model card for its license/terms.
"""

from __future__ import annotations

import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image

# onnxruntime is imported lazily inside load() so that importing this module
# (e.g. for the dataclasses) never hard-fails when the runtime is absent.
_rt = None


# Category ids as they appear in selected_tags.csv -> human label.
CATEGORY_LABELS = {
    0: "general",
    4: "character",
    9: "rating",
}

# Kaomoji tags must never have their underscores rewritten.
_KAOMOJIS = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=",
    ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}

ImageInput = Union[str, Path, Image.Image]


@dataclass(frozen=True)
class Tag:
    name: str
    category: str   # "general" | "character" | "rating" | "other"
    score: float


@dataclass
class TagResult:
    """Structured tagging output for a single image."""
    general: List[Tag] = field(default_factory=list)
    character: List[Tag] = field(default_factory=list)
    rating: List[Tag] = field(default_factory=list)
    other: List[Tag] = field(default_factory=list)

    @property
    def all(self) -> List[Tag]:
        """Every kept tag, character first, then general, then rating/other."""
        return self.character + self.general + self.rating + self.other

    @property
    def predicted_rating(self) -> Optional[str]:
        """Highest-scoring rating tag name, or None if rating is disabled/empty."""
        return self.rating[0].name if self.rating else None

    def names(self) -> List[str]:
        return [t.name for t in self.all]

    def caption(self, escape_parens: bool = False) -> str:
        """Flat comma-joined caption. Set escape_parens=True only if you are
        feeding this back into a Stable Diffusion prompt."""
        names = self.character_names() + self.general_names()
        if escape_parens:
            names = [n.replace("(", r"\(").replace(")", r"\)") for n in names]
        return ", ".join(names)

    def general_names(self) -> List[str]:
        return [t.name for t in self.general]

    def character_names(self) -> List[str]:
        return [t.name for t in self.character]


def _build_providers() -> list:
    """CUDA first (if the installed runtime has it), CPU fallback."""
    available = set(_rt.get_available_providers())
    providers: list = []
    if "CUDAExecutionProvider" in available:
        providers.append((
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ))
    providers.append("CPUExecutionProvider")
    return providers


class WDTagger:
    """Thread-safe WD-EVA02 tagger. Construct once, reuse for the whole process.

    Example
    -------
        tagger = WDTagger("models/wd-eva02-large-tagger-v3")
        result = tagger.tag_image("cat.jpg")
        print(result.predicted_rating, result.names())
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        *,
        gen_threshold: float = 0.35,
        char_threshold: float = 0.85,
        rating_threshold: float = 0.0,
        rating_enabled: bool = True,
        keep_underscores: bool = True,
        target_size: int = 448,
    ):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.onnx"
        self.csv_path = self.model_dir / "selected_tags.csv"

        self.gen_threshold = gen_threshold
        self.char_threshold = char_threshold
        self.rating_threshold = rating_threshold
        self.rating_enabled = rating_enabled
        self.keep_underscores = keep_underscores
        self.target_size = target_size

        self._session = None
        self._tag_names: List[str] = []
        self._categories: List[int] = []
        self._load_lock = threading.Lock()

    # ----- lifecycle -------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def device(self) -> str:
        if not self.is_loaded:
            return "not loaded"
        prov = self._session.get_providers()[0]
        return "GPU (CUDA)" if "CUDA" in prov else "CPU"

    def load(self) -> "WDTagger":
        """Load the vocabulary + ONNX session. Idempotent and thread-safe."""
        if self._session is not None:
            return self
        with self._load_lock:
            if self._session is not None:
                return self

            global _rt
            if _rt is None:
                import onnxruntime as ort  # lazy; clearer error if missing
                _rt = ort

            if not self.model_path.exists() or not self.csv_path.exists():
                raise FileNotFoundError(
                    f"Missing model files in {self.model_dir}. Expected "
                    f"model.onnx and selected_tags.csv. Download them from "
                    f"https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3"
                )

            self._load_vocabulary()
            self._session = _rt.InferenceSession(
                str(self.model_path), providers=_build_providers()
            )
        return self

    def unload(self) -> None:
        """Free the ONNX session. Safe to call repeatedly."""
        with self._load_lock:
            self._session = None
            self._tag_names = []
            self._categories = []

    def _load_vocabulary(self) -> None:
        names: List[str] = []
        cats: List[int] = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = row["name"]
                cat = int(float(row["category"]))
                if self.keep_underscores or raw in _KAOMOJIS:
                    name = raw
                else:
                    name = raw.replace("_", " ")
                names.append(name)
                cats.append(cat)
        self._tag_names = names
        self._categories = cats

    # ----- preprocessing (tuned to the model — do not refactor) ------------

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        image = canvas.convert("RGB")

        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))
        if max_dim != self.target_size:
            padded = padded.resize((self.target_size, self.target_size), Image.BICUBIC)

        arr = np.asarray(padded, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR
        return np.expand_dims(arr, axis=0)

    # ----- inference -------------------------------------------------------

    def tag_image(
        self,
        image: ImageInput,
        *,
        gen_threshold: Optional[float] = None,
        char_threshold: Optional[float] = None,
        rating_threshold: Optional[float] = None,
    ) -> TagResult:
        """Tag a single image (path or PIL.Image) into a structured TagResult."""
        self.load()
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                return self._predict(im, gen_threshold, char_threshold, rating_threshold)
        return self._predict(image, gen_threshold, char_threshold, rating_threshold)

    def _predict(
        self,
        image: Image.Image,
        gen_threshold: Optional[float],
        char_threshold: Optional[float],
        rating_threshold: Optional[float],
    ) -> TagResult:
        gen_t = self.gen_threshold if gen_threshold is None else gen_threshold
        char_t = self.char_threshold if char_threshold is None else char_threshold
        rating_t = self.rating_threshold if rating_threshold is None else rating_threshold

        arr = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        preds = self._session.run(None, {input_name: arr})[0][0]

        result = TagResult()
        for i, score in enumerate(preds):
            if i >= len(self._tag_names):
                break
            cat = self._categories[i]
            label = CATEGORY_LABELS.get(cat, "other")

            if label == "general":
                if score > gen_t:
                    result.general.append(Tag(self._tag_names[i], label, float(score)))
            elif label == "character":
                if score > char_t:
                    result.character.append(Tag(self._tag_names[i], label, float(score)))
            elif label == "rating":
                if self.rating_enabled and score > rating_t:
                    result.rating.append(Tag(self._tag_names[i], label, float(score)))
            else:
                if score > gen_t:
                    result.other.append(Tag(self._tag_names[i], label, float(score)))

        result.general.sort(key=lambda t: t.score, reverse=True)
        result.character.sort(key=lambda t: t.score, reverse=True)
        result.rating.sort(key=lambda t: t.score, reverse=True)
        result.other.sort(key=lambda t: t.score, reverse=True)
        return result

    def tag_paths(
        self,
        paths: Iterable[Union[str, Path]],
        *,
        max_workers: int = 4,
        on_result: Optional[Callable[[Path, Optional[TagResult], Optional[str]], None]] = None,
        **thresholds,
    ) -> Dict[Path, Optional[TagResult]]:
        """Tag many images concurrently. Returns {path: TagResult or None}.

        `on_result(path, result, error)` is called as each image finishes —
        use it to stream progress or write straight into your DB. ONNX
        InferenceSession.run is safe to call from multiple threads.
        """
        self.load()
        results: Dict[Path, Optional[TagResult]] = {}

        def work(p: Union[str, Path]):
            p = Path(p)
            try:
                return p, self.tag_image(p, **thresholds), None
            except Exception as e:  # noqa: BLE001 - report, don't crash the batch
                return p, None, str(e)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(work, p) for p in paths]
            for fut in as_completed(futures):
                path, res, err = fut.result()
                results[path] = res
                if on_result is not None:
                    on_result(path, res, err)
        return results


# ===========================================================================
# OPTIONAL: tiny SQLite store for gallery search.
# Delete this section if TrackImage owns its own schema — the tagger above is
# fully independent of it.
# ===========================================================================
import sqlite3  # noqa: E402


class TagStore:
    """Minimal tag index for Booru-style search over a gallery.

    Stores one row per (image, tag) with its confidence so you can rank,
    filter by category, and AND/OR multiple tags at query time.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS image_tags (
                image_id INTEGER NOT NULL,
                tag      TEXT    NOT NULL,
                category TEXT    NOT NULL,
                score    REAL    NOT NULL,
                PRIMARY KEY (image_id, tag)
            );
            CREATE INDEX IF NOT EXISTS idx_image_tags_tag   ON image_tags(tag);
            CREATE INDEX IF NOT EXISTS idx_image_tags_image ON image_tags(image_id);
            """
        )
        self.conn.commit()

    def save(self, image_id: int, result: TagResult, *, replace: bool = True) -> None:
        cur = self.conn.cursor()
        if replace:
            cur.execute("DELETE FROM image_tags WHERE image_id = ?", (image_id,))
        cur.executemany(
            "INSERT OR REPLACE INTO image_tags(image_id, tag, category, score) "
            "VALUES (?, ?, ?, ?)",
            [(image_id, t.name, t.category, t.score) for t in result.all],
        )
        self.conn.commit()

    def search(self, tags: List[str], *, mode: str = "AND", limit: int = 200) -> List[int]:
        """Return image_ids matching the given tags. mode = 'AND' | 'OR'."""
        if not tags:
            return []
        placeholders = ",".join("?" for _ in tags)
        if mode.upper() == "OR":
            sql = (
                f"SELECT image_id FROM image_tags WHERE tag IN ({placeholders}) "
                f"GROUP BY image_id ORDER BY SUM(score) DESC LIMIT ?"
            )
            params = [*tags, limit]
        else:  # AND: image must carry every requested tag
            sql = (
                f"SELECT image_id FROM image_tags WHERE tag IN ({placeholders}) "
                f"GROUP BY image_id HAVING COUNT(DISTINCT tag) = ? "
                f"ORDER BY SUM(score) DESC LIMIT ?"
            )
            params = [*tags, len(set(tags)), limit]
        return [row[0] for row in self.conn.execute(sql, params)]

    def autocomplete(self, prefix: str, *, limit: int = 20) -> List[str]:
        """Tag suggestions by popularity for a typed prefix."""
        rows = self.conn.execute(
            "SELECT tag, COUNT(*) c FROM image_tags WHERE tag LIKE ? "
            "GROUP BY tag ORDER BY c DESC LIMIT ?",
            (prefix + "%", limit),
        )
        return [row[0] for row in rows]

    def tags_for(self, image_id: int) -> List[Tag]:
        rows = self.conn.execute(
            "SELECT tag, category, score FROM image_tags WHERE image_id = ? "
            "ORDER BY score DESC",
            (image_id,),
        )
        return [Tag(r[0], r[1], r[2]) for r in rows]


# ===========================================================================
# CLI smoke test:  python tagger.py <model_dir> <image_or_folder>
# ===========================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python tagger.py <model_dir> <image_or_folder>")
        raise SystemExit(2)

    model_dir, target = sys.argv[1], Path(sys.argv[2])
    tagger = WDTagger(model_dir).load()
    print(f"loaded on {tagger.device}")

    if target.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        images = [p for p in target.iterdir() if p.suffix.lower() in exts]

        def _print(p, res, err):
            if err:
                print(f"  ! {p.name}: {err}")
            else:
                print(f"  {p.name}  [{res.predicted_rating}]  {', '.join(res.names()[:12])}…")

        tagger.tag_paths(images, on_result=_print)
    else:
        res = tagger.tag_image(target)
        print("rating  :", res.predicted_rating)
        print("character:", res.character_names())
        print("general :", res.general_names())
