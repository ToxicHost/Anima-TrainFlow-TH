# TrackImage auto-tagger (`tagger.py`)

A single-file, drop-in **Booru auto-tagger** for an image gallery, extracted
from the Anima-TrainFlow studio and reshaped for search instead of LoRA
training. It runs SmilingWolf's **WD-EVA02-Large Tagger v3** through ONNX and
returns structured tags + per-tag confidence so you can store them in a DB and
search like a Booru.

This is a handoff artifact — `tagger.py` does **not** import anything from
TrainFlow. Copy the one file into TrackImage and go.

---

## 1. What you get

- `WDTagger` — load the model once, tag images (single or multi-threaded batch).
- `TagResult` / `Tag` — structured output: `general`, `character`, `rating`,
  each with a `score`. No prompt-formatting baggage.
- `TagStore` (optional) — a tiny SQLite index with AND/OR tag search,
  autocomplete, and per-image lookups. Delete it if TrackImage owns its schema.

### How it differs from the trainer's tagger
| | Trainer (TrainFlow) | This module |
|---|---|---|
| Output | one comma-joined caption string | structured `(tag, category, score)` |
| Parentheses | backslash-escaped for SD prompts | raw |
| Underscores | replaced with spaces | **kept** (`long_hair`) — Booru convention |
| Rating tag | ignored | exposed (`predicted_rating`) |
| Storage | `<image>.txt` sidecars | your DB (SQLite helper included) |
| Deps | onnxruntime, numpy, pandas, **torch** | onnxruntime, numpy, pillow |

---

## 2. Install

```bash
pip install onnxruntime        # CPU
# or, for an NVIDIA GPU:
pip install onnxruntime-gpu
```

`numpy` and `pillow` are already in TrackImage. No torch, no pandas.

### Get the model (not bundled — ~1.2 GB)

Download these two files from
<https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3> into a folder:

```
models/wd-eva02-large-tagger-v3/
├── model.onnx
└── selected_tags.csv
```

Quick fetch:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download as d; r='SmilingWolf/wd-eva02-large-tagger-v3'; \
[d(r, f, local_dir='models/wd-eva02-large-tagger-v3') for f in ('model.onnx','selected_tags.csv')]"
```

---

## 3. Use it

### One image
```python
from tagger import WDTagger

tagger = WDTagger("models/wd-eva02-large-tagger-v3").load()
print("device:", tagger.device)            # "GPU (CUDA)" or "CPU"

res = tagger.tag_image("photo.jpg")
print(res.predicted_rating)                # e.g. "general"
print(res.character_names())               # ["hatsune_miku", ...]
print(res.names())                         # every kept tag, ranked
```

### A whole folder (multi-threaded)
```python
from pathlib import Path

images = [p for p in Path("gallery").iterdir() if p.suffix.lower() in {".png", ".jpg", ".webp"}]
results = tagger.tag_paths(images, max_workers=4,
                           on_result=lambda p, r, e: print(p.name, "ok" if r else e))
```

### With the SQLite store (search)
```python
import sqlite3
from tagger import WDTagger, TagStore

tagger = WDTagger("models/wd-eva02-large-tagger-v3").load()
store  = TagStore(sqlite3.connect("trackimage.db"))

# index on import:
store.save(image_id=42, result=tagger.tag_image("photo.jpg"))

# search:
store.search(["1girl", "outdoors"], mode="AND")   # images with BOTH tags
store.search(["cat", "dog"], mode="OR")           # either tag, ranked
store.autocomplete("blu")                          # ["blue_eyes", "blurry", ...]
store.tags_for(42)                                 # this image's tags
```

### CLI smoke test
```bash
python tagger.py models/wd-eva02-large-tagger-v3 some_image.jpg
python tagger.py models/wd-eva02-large-tagger-v3 some_folder/
```

---

## 4. Wiring it into TrackImage

TrackImage already has a **watchdog** auto-sync and a worker pool. The clean
integration is:

1. **Tag on import.** In the "new file detected" path, call
   `tagger.tag_image(path)` once, then `store.save(image_id, result)`.
   Don't re-tag on every search.
2. **One shared `WDTagger`.** Construct it once at startup and reuse it; loading
   is idempotent and thread-safe, so the watchdog threads can share it.
3. **Search/autocomplete endpoints.** Back your gallery's search box with
   `TagStore.search(...)` and `TagStore.autocomplete(...)`.
4. **Back-fill once.** Run `tag_paths()` over the existing library a single time
   to populate the index; new files flow in via the watchdog hook.

Because TrackImage's startup log shows it has no GPU stack, expect CPU
inference (a few hundred ms/image at 448px — fine for a personal gallery). For a
large back-fill, install `onnxruntime-gpu` on a machine with CUDA and the same
code uses the GPU automatically.

---

## 5. Tuning knobs (`WDTagger(...)`)

| arg | default | meaning |
|---|---|---|
| `gen_threshold` | `0.35` | min confidence for general tags |
| `char_threshold` | `0.85` | min confidence for character tags |
| `rating_threshold` | `0.0` | min confidence for rating tags (0 = keep all four) |
| `rating_enabled` | `True` | include safe/sensitive/questionable/explicit |
| `keep_underscores` | `True` | `long_hair` vs `long hair` |
| `target_size` | `448` | model input size — **do not change** |

For search it's usually better to tag at a *lower* general threshold (e.g.
`0.25`) and keep the `score`, then filter/rank at query time rather than
throwing tags away up front. Per-call overrides are supported:
`tagger.tag_image(path, gen_threshold=0.25)`.

---

## 6. Notes & credit

- The ONNX preprocessing (square white-pad → 448 → BGR) is tuned to this exact
  model. Leave it alone.
- Model: **wd-eva02-large-tagger-v3** by **SmilingWolf**. Review its model card
  for license/terms before redistributing weights.
- `tagger.py` is self-contained; the `TagStore` section is optional and clearly
  fenced off if you'd rather use TrackImage's own database layer.
