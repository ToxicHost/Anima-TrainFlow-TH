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

## VRAM Preset selector
- Added a **"VRAM Preset"** dropdown (Custom / 6 GB / 12 GB / 16 GB / 24 GB). Selecting a card
  populates existing controls — Network Rank, Blocks to Swap, Batch Size, preview width/height,
  and the bucketing crop targets — with per-card starting points. **"Custom" is a no-op**, so it
  changes nothing by default.
- It is a pure convenience: it only drives controls that already exist and are already persisted
  (no new training parameters, no change to defaults). The preset itself is not persisted; on
  reload it shows "Custom" while the underlying values persist.
- Two caveats are shown in the info line: block-swap values are starting points (raise on OOM,
  lower if too slow), and the resolution targets only take effect after re-running **Smart Aspect
  Ratio Bucketing**. Presets target the LoRA path; with Full Fine-tune on, raise block-swap
  manually (full-FT needs far more VRAM).

## In-App Updater
- Added a **"Check for Updates"** button that fetches the latest `app.py` from the `main` branch.
- It downloads to memory, validates the payload, compares versions, backs up the current file to
  `app.py.bak`, then atomically swaps in the new file via a temp file. It prompts the user to
  **restart** to apply (a running app cannot hot-swap its own code).
- It touches **only** `app.py` / `app.py.bak` / a temp file in the app directory — never
  `settings.json`, models, the runtime, datasets, or training output. On any download/validation
  failure the current `app.py` is left completely untouched.

## Repo hygiene
- Added a root `.gitignore` covering the portable runtime, model weights, training
  outputs/caches, and per-user state (`settings.json`).
