# Modifications

This is an MIT-licensed fork of [ThetaCursed/Anima-TrainFlow](https://github.com/ThetaCursed/Anima-TrainFlow).
The original LICENSE and in-app attribution ("Created by ThetaCursed") are preserved.

The changes below surface training capabilities that already exist in the bundled
trainer (`training/sd-scripts/`) but were not exposed in the GUI, plus a caption
tag-pruning utility. **All new controls are opt-in: leaving them untouched
reproduces upstream behavior exactly.**

## Block-swap support (`blocks_to_swap`)
- Added a **"Blocks to Swap (0 = off)"** slider (0–26) next to the training controls.
- The value is threaded through `create_training_toml()` and written into the training
  config **only when > 0**, so the default config is byte-identical to upstream.
- Clamped to the trainer's supported range (`0 .. num_blocks - 2 = 26`), with a log note
  when a larger value is entered.
- The underlying offload implementation
  (`library/custom_offloading_utils.py`, `library/anima_models.enable_block_swap`) is
  unchanged — this only wires the existing `--blocks_to_swap` argument through the GUI.

## Caption tag-pruning tool
- Added a **"Prune Tags"** section: a comma-separated textbox + "Strip Tags from Captions"
  button. Useful for character/concept LoRAs where identity tags should be absorbed by the
  trigger word.
- `run_prune_tags()` backs up all `*.txt` captions to a sibling `captions_backup/` (once,
  if absent), then removes each tag whole-word and case-insensitively
  (`\b<tag>\b`, `re.IGNORECASE`) and normalizes the result (no `, ,` artifacts, no stray
  leading/trailing commas). Reports files-changed and tag-occurrence counts.

## Full fine-tune toggle (opt-in)
- Added a **"Full Fine-tune (no LoRA)"** checkbox. When enabled, training is routed to the
  existing full-DiT trainer `anima_train.py` instead of `anima_train_network.py`.
- `create_full_finetune_toml()` builds a config from only the keys `anima_train.py` accepts
  — **no `network_*` keys**. The LLM adapter is explicitly kept frozen
  (`llm_adapter_lr=0.0`), precision stays bf16, and `blocks_to_swap` is honored.
- When the checkbox is on, the LoRA **Network Rank** field is made inert, and the log warns
  that full fine-tune needs far more VRAM (recommend block-swap on smaller cards).

## Anima training constraints honored
- **bf16 only** — no fp16 path added anywhere.
- **LLM adapter stays frozen** (LoRA: `network_train_unet_only=True`; full-FT:
  `llm_adapter_lr=0.0`).
- **No fp8-DiT training** option added.
- Block-swap capped at 26 (`num_blocks - 2`).

## Repo hygiene
- Added a root `.gitignore` covering the portable runtime, model weights, training
  outputs/caches, and per-user state (`settings.json`).
