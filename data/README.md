# Data directory

This directory contains **data splits and metadata** for the CIC-Intents
study. Raw text of the four channel corpora is **not redistributed**
here — see [`../docs/data_sources.md`](../docs/data_sources.md) for
download instructions.

## Contents
data/
├── LICENSE # CC-BY-4.0 for redistributed artefacts
├── README.md # this file
├── raw/ # not committed — download via docs/data_sources.md
├── splits/ # train/val/test splits + test-set predictions
│ ├── main_preds.npz
│ ├── loo_forum_preds.npz
│ └── README.md
└── corpus.csv # (optional) processed corpus metadata, no raw text

text

## What's redistributed here

| File | Description | Size |
|---|---|---|
| `splits/main_preds.npz` | Test-set predictions + ground-truth labels, main split (N=143) | ~50 KB |
| `splits/loo_forum_preds.npz` | Test-set predictions on all 113 forum documents (leave-forum-out) | ~40 KB |
| `corpus.csv` | Per-document metadata: channel, use_mention, intent labels, markers. **No raw text.** | ~200 KB |

## What's NOT redistributed

- Raw email / SMS / Telegram / forum text — governed by source datasets.
  See `../docs/data_sources.md` for download instructions.
- Trained model checkpoints (`.pt` files, ~3.5 GB total) — hosted
  externally on Kaggle. See `../models/README.md`.

## Quick check

After downloading the four corpora (see `docs/data_sources.md`),
regenerate the splits with:

```bash
python scripts/build_splits.py --seed 42 --out data/splits/
Expected output: main_preds.npz, loo_forum_preds.npz, and the
five forum CV fold files. All random operations are seeded with 42.

License
Redistributed artefacts are under CC-BY-4.0 — see LICENSE.
Original text retains the license of each source corpus.
