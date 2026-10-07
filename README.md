# CIC-Intents: Channel-Invariant, Interpretable and Causally Grounded Intent Detection for Italian Multi-Channel Fraud

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

Official code, data splits, and reproducibility artefacts for the paper:

> **CIC-Intents: Channel-Invariant, Interpretable and Causally Grounded Intent Detection for Italian Multi-Channel Fraud**
> Mariia Bilikhodze
> University of Pavia, Department of Economics and Management
> Submitted to *Information Processing & Management* (Elsevier), 2026.

---

## TL;DR

Fine-tuned Italian BERT (UmBERTo) reaches **99.75%** in-domain accuracy on phishing detection but collapses to **2.43%** on held-out SMS under leave-one-channel-out validation. The encoder learns *channel* rather than *intent*. **CIC-Intents** fixes this by combining four components:

1. **IFIT** — Italian Fraud Intent Taxonomy (12 intents, multi-label).
2. **Multi-Label SupCon** — contrastive loss with cross-channel weighting.
3. **Kernel MMD + PCC** — distribution alignment during training + post-hoc linear centering.
4. **Gradient-based causal attribution** — PN/PS per token.

The system reaches **mAP 0.723** and **Top-5 recall 0.924**, with balanced email and SMS performance. Ablation shows PCC alone accounts for the -54.7% reduction in Domain Mutual Information (0.739 → 0.335) without damaging the intent signal.

---

## Repository structure

```
cic-intents/
├── data/               # Annotated corpus (950 docs) + intermediate splits
├── docs/               # Annotation prompt, taxonomy, pipeline walkthrough
├── models/             # Model checkpoints (not committed; see models/README.md)
├── notebooks/          # Full end-to-end pipeline notebook
├── paper/              # LaTeX source, bibliography, PDF
├── results/            # Metric CSVs and figures
├── src/                # Reusable Python modules
├── requirements.txt
├── CITATION.cff
├── LICENSE             # MIT (code)
└── README.md
```

---

## Quick start

### 1. Install

```bash
git clone https://github.com/marybilik/cic-intents.git
cd cic-intents
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download it_core_news_sm
```

### 2. Reproduce headline metrics (no GPU needed for reading)

The final annotated corpus `data/df_cicl.csv` (950 documents, 12 intents, 3 channels) is committed to the repo. All headline metrics in the paper can be recomputed from it.

```python
import pandas as pd
df = pd.read_csv('data/df_cicl.csv')
print(df['channel_id'].value_counts())
# 0 (email): 421,  1 (sms): 416,  3 (forum): 113
```

### 3. Retrain CIC-Intents end-to-end (GPU recommended)

Open `notebooks/cic_intents_full_pipeline.ipynb`. It is self-contained:

- **Part A** (Colab, internet): data acquisition + preprocessing + annotation.
- **Part B** (Kaggle / local GPU): training + evaluation + figures.

Training takes ~40 minutes on a single NVIDIA T4 for the main model, ~30 minutes for LOO-forum. The λ-MMD sweep is ~3 hours on the same hardware.

### 4. Reproduce individual figures

```bash
# All figures from a trained checkpoint:
python -m src.figures --checkpoint models/cicl_main.pt --outdir results/figures/

# Post-hoc channel centering analysis (Table 8, Table 12):
python -m src.pcc --checkpoint models/cicl_main.pt --data data/df_cicl.csv

# Causal attribution (Table 14, Figure 10):
python -m src.attribution --checkpoint models/cicl_main.pt --data data/df_cicl.csv
```

---

## The Italian Fraud Intent Taxonomy (IFIT)

12 intents, multi-label, in two families:

**Explicit requests (5):** `credential_request`, `payment_request`, `data_request`, `click_request`, `call_request`.

**Implicit manipulations (7):** `urgency`, `authority`, `fear`, `greed`, `impersonation`, `social_proof`, `reciprocity`.

Mean intents per fraud document: **2.97**. Full definitions in `docs/taxonomy.md`.

---

## Datasets

| Channel | Total | Fraud | Non-fraud | Role |
|---|---|---|---|---|
| Email (E-PhishLLM) | 2,701 | 1,131 | 1,570 | use / legit |
| SMS (Smishing-IMC25) | 536 | 536 | 0 | use |
| Telegram (MuLTa) | 827 | 0 | 827 | mention |
| Forum (scraped) | 312 | 0 | 312 | mention |
| **Total** | **4,376** | 1,667 | 2,709 | |

Annotated subset used for intent training: **950 documents** (email fraud + email legit + SMS fraud + forum), interleaved across all three channels.

Raw corpora are pulled from:
- **E-PhishLLM** — `pajola/e-phishGen` on HuggingFace.
- **Smishing-IMC25** — `reportsmishing/Smishing-Dataset-IMC25` on GitHub.
- **MuLTa-Telegram** — `dhfbk/MuLTa-Telegram` on GitHub.
- **Forum** — scraped from Digital-Forum and MilanWorld (script in `src/`).

See `data/README.md` for the full provenance and licensing.

---

## Reproducibility checklist

| Item | Value |
|---|---|
| Random seed | 42 (all experiments) |
| Hardware | NVIDIA T4 (Kaggle) for training; CPU for post-hoc |
| Optimizer | AdamW, lr = 5e-5, weight decay 0.01 |
| Batch size | 32 |
| Max sequence length | 160 |
| Epochs | 10 |
| Encoder | `Musixmatch/umberto-commoncrawl-cased-v1` |
| τ (SupCon temperature) | 0.07 |
| α (channel weight) | 0.5 |
| λ_SupCon | 0.3 |
| λ_MMD | 0.3 (elbow of Pareto frontier) |
| pos_weight clipping | 5 |
| Bootstrap resamples | 1000 |
| Attribution top-k | {5, 10, 20} |

Full details in `paper/main.pdf`, Appendix A.

---

## License

- **Code** (`src/`, `notebooks/`): MIT — see `LICENSE`.
- **Paper** (`paper/`): CC BY 4.0 — see `paper/LICENSE`.
- **Datasets** (`data/`): combination of upstream licenses. See `data/README.md` for attribution requirements of E-PhishLLM, Smishing-IMC25, and MuLTa-Telegram.

---

## Citation

```bibtex
@article{bilikhodze2026cicintents,
  title   = {{CIC-Intents}: Channel-Invariant, Interpretable and Causally
             Grounded Intent Detection for Italian Multi-Channel Fraud},
  author  = {Bilikhodze, Mariia},
  journal = {Information Processing \& Management},
  year    = {2026},
  note    = {Under review}
}
```

See `CITATION.cff` for a machine-readable version.

---

## Contact

Issues and questions: please open a GitHub issue or email `mariia.bilikhodze01@universitadipavia.it`.

## Acknowledgements

UmBERTo model: Musixmatch Research. spaCy Italian pipeline: Explosion AI. Compute: Kaggle T4 GPU program.
