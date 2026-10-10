# Notebooks

Two notebooks reproduce the full pipeline end-to-end. Total runtime on
Kaggle (NVIDIA T4 GPU): **~4 hours** for both, including training of
all models and post-hoc analyses.

## `01-corpus-construction.ipynb`

Corpus build + Gemini annotation. Requires internet access for
HuggingFace downloads, `git clone`, forum scraping, and the Gemini API.

### Contents

| Cell | Purpose |
|---|---|
| 1. Env guards | `TOKENIZERS_PARALLELISM`, warnings |
| 2. Install | `datasets`, `huggingface_hub`, `cloudscraper`, `spacy`, `beautifulsoup4`, `it_core_news_sm` |
| 3. Setup | Imports, paths, seed 42, IFIT codes, channel map |
| 4. Forum scraping | Digital-Forum + MilanWorld via `cloudscraper` (XenForo parser) |
| 5. HuggingFace corpora | E-PhishLLM, Smishing-IMC25, MuLTa-Telegram |
| 6. Raw EDA | Missing texts, duplicates, length stats, cross-source dup check |
| 7. Preprocessing v7 | Strict HTML, entity tags, URL/email/phone, homoglyphs, spaCy lemmatization, markers |
| 8. IFIT annotation | Zero-shot Gemini (`gemini-flash-latest` auto-discovered), checkpoint every 25 docs |
| 9. Build final corpus | Multi-hot matrix, `channel_id`, writes `df_cicl.csv` (950 rows, 12 intents) |

### Running

**On Kaggle** (recommended):

1. Create a new Notebook.
2. Upload `01-corpus-construction.ipynb`.
3. **Add data:** none required — all sources are fetched at runtime.
4. Set accelerator: **None** (this notebook is CPU-only).
5. Enable internet.
6. Set the `GEMINI_API_KEY` secret in **Add-ons → Secrets**.
7. Run all.

**Locally:**

```bash
pip install -r ../requirements.txt
python -m spacy download it_core_news_sm
export GEMINI_API_KEY="..."
jupyter lab 01-corpus-construction.ipynb
```

**Output:** `df_cicl.csv` (950 rows, 12-intent multi-hot labels),
`forum_corpus_max.csv`, `df_lab.csv`, `df_f.csv`,
`annotation_checkpoint.csv`, `annotation_full.csv`.

---

## `02-training-evaluation.ipynb`

Training, evaluation, bootstrap, PCC, PN/PS, and all paper figures.
Requires `df_cicl.csv` from notebook 01.

### Contents

| Cell | Purpose |
|---|---|
| 1. Setup | Imports, seeds, GPU check, IFIT codes |
| 2. Load corpus | `df_cicl.csv`, parse multi-hot, add `n_intents` |
| 3. Build splits | Main 70/15/15 + LOO-forum + 5-fold CV (`src.splits.make_splits`) |
| 4. Model | `CICLMMDv2` (UmBERTo + intent head + projection + MMD) |
| 5. Training utilities | `train_cicl`, `predict_proba`, `get_cls_embeddings`, `channel_center`, `domain_mi_cv` |
| 6. Main model | Trains `cicl_main.pt` (email + SMS + forum) |
| 7. LOO-forum model | Trains `cicl_loo.pt` (email + SMS only) |
| 8. Main evaluation | mAP, AUC, Top-k, per-channel, per-intent |
| 9. LOO-forum evaluation | Three-regime forum comparison (hold-out / CV / LOO) |
| 10. Encoder comparison | UmBERTo vs XLM-R vs mDeBERTa |
| 11. λ-sweep | Pareto frontier (Macro-F1 vs Domain MI) |
| 12. Ablation study | Five component configurations |
| 13. Bootstrap CIs | 95% CIs + paired tests |
| 14. PCC | Raw vs centered CLS analysis (train-means → test) |
| 15. Causal attribution | Gradient×input saliency, PN/PS sweep |
| 16. Figures | Generates all figures in `paper/figures/` |
| 17. Export | Writes `results/*.json`, `results/*.csv` |

### Running

**On Kaggle** (recommended):

1. Create a new Notebook.
2. Upload `02-training-evaluation.ipynb`.
3. **Add data:**
   - Dataset `mariabilik/df-cicl` (required — output of notebook 01)
   - Dataset `mariabilik/checks` (optional — pre-trained checkpoints)
4. Set accelerator: **GPU T4 ×1** or **P100**.
5. Enable internet (needed for HuggingFace downloads).
6. Run all.

**Locally:**

```bash
pip install -r ../requirements.txt
jupyter lab 02-training-evaluation.ipynb
```

Requires a GPU with ≥ 15 GB VRAM for training. Post-hoc analyses
(bootstrap, causal attribution, figures) run on CPU.

### Environment variables

```bash
export CIC_DATA_DIR="./data"        # where df_cicl.csv lives
export CIC_CKPT_DIR="./models"      # where checkpoints are read/written
export CIC_OUT_DIR="./results"      # where outputs are written
export KAGGLE_USERNAME="marybilik"  # only if using kaggle CLI
export KAGGLE_KEY="..."
```

## Notes

- All random operations use seed **42**.
- Checkpoints are written every epoch to `/kaggle/working/`.
- Download the outputs via the Kaggle Output panel before the session ends
  — Kaggle wipes `/kaggle/working/` after ~12h of inactivity.
- See [`../docs/reproducibility.md`](../docs/reproducibility.md) for the full
  checklist.
- Two PCC protocols are reported in the paper: the headline number uses
  **train-means → test** (0.676 → 0.116, −82.9%); the ablation uses
  **test-means** (0.676 → 0.442 for the full model; 0.788 → 0.380 for
  BCE-only). Both are documented in
  [`../docs/reproducibility.md`](../docs/reproducibility.md).
