# Data directory

This directory contains **data splits and metadata** for the CIC-Intents
study. Raw text of the four channel corpora is **not redistributed**
here — see [`../docs/data_sources.md`](../docs/data_sources.md) for
download instructions.

## Contents

```
data/
├── LICENSE                 # CC-BY-4.0 for redistributed artefacts
├── README.md               # this file
├── raw/                    # not committed — download via docs/data_sources.md
├── splits/                 # train/val/test splits + test-set predictions
│   ├── main_preds.npz
│   ├── loo_forum_preds.npz
│   └── README.md
└── corpus.csv              # processed corpus metadata, no raw text
```

## What is redistributed here

| File | Description | Size |
|---|---|---|
| `splits/main_preds.npz` | Test-set predictions + ground-truth labels, main split (`N = 143`) | ~50 KB |
| `splits/loo_forum_preds.npz` | Test-set predictions on all 113 forum documents (leave-forum-out) | ~40 KB |
| `corpus.csv` | Per-document metadata: channel, use/mention, intent labels, markers. **No raw text.** | ~200 KB |

## What is NOT redistributed

- **Raw email / SMS / Telegram / forum text** — governed by the source
  datasets. See [`../docs/data_sources.md`](../docs/data_sources.md) for
  download instructions.
- **Trained model checkpoints** (`.pt` files, ~3.5 GB total) — hosted
  externally on Kaggle. See [`../models/README.md`](../models/README.md).

## Quick check

The splits are regenerated from the annotated corpus `df_cicl.csv`
by [`../notebooks/02-training-evaluation.ipynb`](../notebooks/02-training-evaluation.ipynb):

```python
import pandas as pd
from src.splits import make_splits

df = pd.read_csv('data/df_cicl.csv')
splits = make_splits(df)

# splits['main']['test']  -- 143 docs
# splits['loo']['test']   -- all 113 forum
# splits['cv']['folds']   -- 5 stratified folds
```

Expected output:

- `main` split — 665 train / 142 val / 143 test, stratified by
  `(channel_id, has_any_intent)`.
- `loo` split — 711 train / 126 val, test = all 113 forum documents.
- `cv` split — 5 folds, 113 forum documents total.

All random operations are seeded with `RANDOM_STATE = 42`.

## License

Redistributed artefacts are under **CC-BY-4.0** — see [`LICENSE`](LICENSE).
Original raw text retains the license of each source corpus.
