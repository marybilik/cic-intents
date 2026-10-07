# CIC-Intents: Channel-Invariant, Interpretable and Causally Grounded Intent Detection for Italian Multi-Channel Fraud

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/paper-preprint-blue)](paper/main.pdf)

Official code, data splits, and reproducibility artefacts for the paper:

> **CIC-Intents: Channel-Invariant, Interpretable and Causally Grounded Intent Detection for Italian Multi-Channel Fraud**
> Mariia Bilikhodze
> University of Pavia, Department of Economics and Management
> Submitted to *Information Processing & Management* (Elsevier), 2026.

---

## TL;DR

A fine-tuned Italian BERT encoder (UmBERTo) reaches **99.75%** accuracy in-domain but collapses to **2.43%** on held-out SMS under leave-one-channel-out (LOO) validation — a **41×** degradation. We call this the *channel shortcut*: the model has learned the surface form of the training channel rather than the semantics of manipulation.

We propose **CIC-Intents**, a framework that combines:

1. **IFIT** — a 12-intent multi-label taxonomy for Italian fraud (5 explicit requests + 7 implicit manipulations).
2. **Multi-Label Supervised Contrastive learning** — positive pairs formed by intent overlap, with higher weight for cross-channel pairs.
3. **Kernel MMD regularizer** on the CLS embedding, followed by **post-hoc channel centering** (the decisive component).
4. **Gradient-based causal attribution** using Probability of Necessity and Sufficiency.

## Key results

| Metric | Value | 95% CI (bootstrap, n=1000) |
|---|---|---|
| mAP (macro, 9 intents) | **0.723** | [0.640, 0.806] |
| Macro-F1 (threshold 0.5) | 0.464 | [0.412, 0.504] |
| Top-5 recall | **0.924** | [0.894, 0.950] |

**Per-channel macro-F1:** Email 0.452 · SMS 0.453 · Forum 0.151–0.255 (three regimes).

**Encoder comparison (identical framework, identical splits):**

| Encoder | Params | Macro-F1 | mAP | Top-5 |
|---|---|---|---|---|
| **UmBERTo** | 125M | 0.464 | **0.723** | **0.924** |
| XLM-R (base) | 278M | 0.474 | 0.682 | 0.900 |
| mDeBERTa-v3 (base) | 278M | **0.482** | 0.673 | 0.907 |

The smallest, Italian-pretrained encoder achieves the strongest ranking metrics — larger multilingual models overfit channel-specific artefacts.

**Post-hoc channel centering** reduces Domain Mutual Information from 0.739 to 0.335 (−54.7%) while leaving k-NN intent accuracy essentially intact (mean |Δ| = 0.009). An ablation shows that centering is the single decisive component: training-time objectives (SupCon, MMD) do not reduce Domain MI below the BCE-only baseline.

## Repository structure
cic-intents/
├── paper/ # LaTeX source, bibliography, compiled PDF
│ ├── main.tex
│ ├── refs.bib
│ ├── main.pdf
│ ├── LICENSE # CC-BY-4.0 for the manuscript
│ └── figures/ # All 11 figures + architecture diagram
├── notebooks/ # Jupyter/Kaggle pipeline (fully reproducible)
│ └── cic_intents_full_pipeline.ipynb
├── data/ # Data splits (no raw text, see data/README.md)
│ ├── LICENSE # CC-BY-4.0 for data
│ ├── splits/
│ │ ├── main_preds.npz # Test-set predictions, main split
│ │ ├── loo_forum_preds.npz # Leave-forum-out predictions
│ │ └── README.md
│ └── README.md
├── results/ # Machine-readable metrics
│ ├── paper_metrics.json
│ ├── forum_cv_results.csv
│ └── README.md
├── docs/ # Extended documentation
│ ├── ifit_taxonomy.md
│ ├── annotation_prompt.md
│ ├── data_sources.md
│ └── reproducibility.md
├── models/ # Trained checkpoints (via external link)
│ └── README.md
├── LICENSE # MIT for code
├── CITATION.cff
├── requirements.txt
└── README.md


## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/marybilik/cic-intents.git
cd cic-intents

2. Install dependencies
bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
3. Download the data
The four channel datasets are publicly available (see docs/data_sources.md). We do not redistribute the raw text — only the exact splits and test-set predictions used in the paper:

bash
# Place the following files in data/splits/:
#   main_preds.npz
#   loo_forum_preds.npz
4. Reproduce the experiments
Open the notebook:

bash
jupyter lab notebooks/cic_intents_full_pipeline.ipynb
The notebook is self-contained and reproduces:

Data loading and preprocessing

Multi-Label SupCon + kernel MMD training

Encoder comparison (UmBERTo, XLM-R, mDeBERTa)

Ablation study (5 configurations)

Bootstrap CIs and paired tests

Pareto frontier analysis

Post-hoc channel centering

Gradient-based causal attribution (PN/PS)

All 11 figures in paper/figures/

Reproducibility: random seed 42 for all experiments; hardware NVIDIA T4 (Kaggle). Full checklist in docs/reproducibility.md.

Data sources
Channel	Source	Role	Type
Email	E-PhishGen (Pajola et al., 2025)	use	LLM-generated
SMS	Smishing-IMC25 (Agarwal et al., 2025)	use	real, user reports
Telegram	MuLTa-Telegram (Leonardelli et al., 2025)	mention	non-hateful subset
Forum	Scraped from Digital-Forum, MilanWorld	mention	annotated by us
See docs/data_sources.md for full provenance, preprocessing, and license notes.

The Italian Fraud Intent Taxonomy (IFIT)
12 intents organized into two families. Full definitions with examples in docs/ifit_taxonomy.md.

Explicit (5): credential_request, payment_request, data_request, click_request, call_request
Implicit (7): urgency, authority, fear, greed, impersonation, social_proof, reciprocity

Mean number of intents per fraud document: 2.97.

Citation
If you use this work, please cite:

bibtex
@article{bilikhodze2026cicintents,
  title  = {{CIC-Intents}: Channel-Invariant, Interpretable and Causally
            Grounded Intent Detection for Italian Multi-Channel Fraud},
  author = {Bilikhodze, Mariia},
  journal = {Information Processing \& Management},
  year   = {2026},
  note   = {Under review}
}
Machine-readable citation: CITATION.cff.

Ethics and data use
Email (E-PhishGen) and SMS (Smishing-IMC25) corpora contain real or LLM-generated phishing messages. We use them exclusively for defensive research (fraud detection). All PII in SMS is pre-anonymized by the original corpus.

Forum posts were scraped from public forums and manually annotated. No personal information is redistributed — only the processed text_clean field appears in aggregated metrics.

Telegram (MuLTa) is a public hate-speech corpus; we use only its non-hateful Italian subset.

The trained models are released for research use only. Deployment in a real fraud-detection pipeline requires re-validation on the target distribution and appropriate human oversight.

Limitations
The email corpus is LLM-generated; real Italian phishing distributions may differ. The Telegram subset is a proxy, not a general non-fraud corpus. Forum evaluation covers only two Italian forums. Results are single-seed; multi-seed verification on a sub-experiment showed σ ≈ 0.01 in macro-F1. See the paper for full discussion.

License
Code (notebooks, scripts, configuration): MIT

Paper, data splits, results: CC-BY-4.0

Contact
Mariia Bilikhodze — University of Pavia, Department of Economics and Management
GitHub: @marybilik
ORCID: [0009-0000-6968-746X](https://orcid.org/0009-0000-6968-746X)
Email: mariia.bilikhodze01@universitadipavia.it
