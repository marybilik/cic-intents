# CIC-Intents

**Channel Shortcuts in Italian Multi-Channel Fraud Intent Detection.**

Code, prompts, data splits and reproduction scripts for the paper
(M. Bilikhodze, University of Pavia).

---

## Repository structure

```text
cic-intents/
├── data/                        # data splits and metadata (no raw text)
│   ├── LICENSE                  # CC-BY-4.0 for redistributed artefacts
│   ├── README.md                # data directory guide
│   ├── raw/                     # not committed — download via docs/data_sources.md
│   └── splits/                  # train/val/test predictions
│       ├── main_preds.npz
│       └── loo_forum_preds.npz
├── docs/                        # paper-level documentation
│   ├── annotation_prompt.md     # exact Gemini prompt + parser
│   ├── data_sources.md          # provenance of all four channels
│   ├── pipeline.md              # end-to-end description
│   ├── reproducibility.md       # step-by-step checklist
│   └── taxonomy.md              # Italian Fraud Intent Taxonomy (IFIT)
├── models/                      # checkpoints (hosted externally)
│   └── README.md
├── notebooks/                   # reproduction notebooks
│   ├── README.md
│   ├── 01-corpus-construction.ipynb
│   └── 02-training-evaluation.ipynb
├── paper/                       # LaTeX source and compiled artefacts
│   ├── LICENSE                  # CC-BY-4.0 for paper source
│   ├── main.tex
│   ├── main.pdf
│   ├── refs.bib
│   ├── research-summary.pdf
│   └── figures/                 # all 13 figures referenced by main.tex
├── results/                     # metrics, predictions, CSVs
│   ├── README.md
│   ├── paper_metrics.json       # canonical metrics
│   ├── ablation_study.csv
│   ├── baseline_encoders.csv
│   ├── bootstrap_ci.json
│   ├── causal_attribution.csv
│   ├── errors_fixed.csv
│   ├── forum_cv_results.csv
│   ├── lambda_sweep.csv
│   └── paired_tests.json
├── src/                         # modular pipeline
│   ├── __init__.py
│   ├── attribution.py           # PN/PS gradient×input saliency
│   ├── cleaning.py              # preprocessing v7 (HTML, homoglyphs, spaCy lemmas)
│   ├── features_v4.py           # 33 hand-crafted intent features
│   ├── figures.py               # all paper figures
│   ├── losses.py                # MultiLabelSupConLoss + channel_mmd_loss
│   ├── model.py                 # CICLMMDv2 (UmBERTo + intent head + projection + MMD)
│   ├── pcc.py                   # Post-hoc Channel Centering + domain MI
│   ├── splits.py                # 70/15/15 + LOO-forum + 5-fold CV
│   ├── train.py                 # train_cicl + predict_proba + get_cls_embeddings
│   └── utils.py                 # seeds, paths, IFIT codes, parsing
├── .gitignore
├── CITATION.cff
├── LICENSE                      # MIT for code
├── README.md
└── requirements.txt
Install
bash
git clone https://github.com/marybilik/cic-intents
cd cic-intents
pip install -r requirements.txt
python -m spacy download it_core_news_sm
Requires a GPU with ≥ 15 GB VRAM for training (T4 / P100).
Post-hoc analyses (bootstrap, causal attribution, figures) run on CPU.

Data
The four-channel Italian corpus is not shipped in this repository.
It is assembled from four independently collected sources, and one of
them (forum) is scraped and requires network access. See
docs/data_sources.md for provenance and licensing.

The final annotated corpus is exported as a single file df_cicl.csv
(950 rows, 12-intent multi-hot labels). It is available on request and as
part of the Kaggle dataset mariabilik/df-cicl.

To rebuild the corpus from scratch, use the notebook
notebooks/01-corpus-construction.ipynb:

bash
export GEMINI_API_KEY="..."          # annotation (Google Gemini)
export CIC_OUT_DIR="./"              # where intermediate CSVs are written
# then run notebooks/01-corpus-construction.ipynb end-to-end
The exact annotation prompt and parser are in
docs/annotation_prompt.md.

Training & evaluation
Trained checkpoints are hosted externally — see
models/README.md. If the checkpoints are present at
$CIC_CKPT_DIR (default ./models/), training is skipped and evaluation
runs directly from them.

Otherwise, to train from scratch:

bash
export CIC_DATA_DIR="./data"         # where df_cicl.csv lives
export CIC_OUT_DIR="./results"
python -m src.train                  # train_cicl on main split
python -m src.pcc                    # Post-hoc Channel Centering + domain MI
python -m src.attribution            # PN/PS causal attribution
python -m src.figures                # generate all paper figures
Environment variables
Variable	Purpose	Default
CIC_DATA_DIR	Where df_cicl.csv lives	./data/
CIC_CKPT_DIR	Where checkpoints live	./models/
CIC_OUT_DIR	Where outputs are written	./results/
CIC_BUILD_CORPUS	1 to re-scrape + re-annotate	0
GEMINI_API_KEY	Gemini key for annotation	—
Frozen configuration
All reported numbers use the following configuration. Deviations are
documented in the paper's Appendix.

python
MODEL_NAME        = "Musixmatch/umberto-commoncrawl-cased-v1"
MAX_LEN           = 160
BATCH_SIZE        = 32
LR                = 5e-5
WEIGHT_DECAY      = 0.01
EPOCHS            = 10
TEMPERATURE       = 0.07       # SupCon temperature
CHANNEL_WEIGHT    = 0.5        # α in Eq. (3)
LAMBDA_SUPCON     = 0.3
LAMBDA_MMD        = 0.3        # final; see λ-sweep for the full curve
POS_WEIGHT_CLIP   = 5.0        # BCE pos_weight clipped to [1, 5]
RANDOM_STATE      = 42
Main results
Held-out test set, main split, N_test = 143.

Metric	Value
mAP (macro, 9 intents, support ≥ 2)	0.721
Mean AUC	0.802
Top-1 recall	0.275
Top-2 recall	0.502
Top-3 recall	0.680
Top-5 recall	0.931
Precision (t = 0.5)	0.368
Recall (t = 0.5)	0.665
Macro-F1 (t = 0.5)	0.465
Exact match	0.133
Per-channel macro-F1:

Channel	Macro-F1	mAP
Email	0.451	0.758
SMS	0.452	0.643
Forum (hold-out, n = 17)	0.136	0.555
Forum — three independent estimates:

Regime	n	Macro-F1	mAP
Hold-out	17	0.136	—
5-fold CV	113	0.270 ± 0.064	0.504 ± 0.082
Leave-forum-out	113	0.228	0.338
Post-hoc Channel Centering (raw CLS):

Metric	Raw CLS	Centered CLS
Domain MI	0.676	0.116
MI reduction	—	−82.9%
k-NN intent (9 intents)	0.843	0.839
|Δ|	—	≈ 0.011
Gradient-based causal attribution:

k	Mean PN	Mean PS
5	0.207	0.168
10	0.294	0.232
20	0.320	0.252
Encoder comparison under identical framework:

Encoder	Params	Macro-F1	mAP	Top-5
UmBERTo	125M	0.465	0.721	0.931
XLM-R (base)	278M	0.460	0.696	0.904
mDeBERTa-v3 (base)	278M	0.501	0.687	0.907
Bootstrap 95% CI (1000 resamples):

Metric	Point estimate	95% CI
mAP (macro)	0.721	[0.645, 0.806]
Macro-F1	0.465	[0.419, 0.504]
Top-5 recall	0.931	[0.901, 0.957]
Full per-intent and per-channel numbers are in
results/paper_metrics.json and in the paper.

Reproducibility
See docs/reproducibility.md for the full checklist.
Summary:

Random seed 42 (numpy, torch, sklearn splits).

Frozen hyperparameters (see table above).

Splits: main 70/15/15 (N_test = 143), LOO-forum (n = 113), 5-fold CV on forum.

Bootstrap CIs: 1000 resamples, seed 42.

Paired tests: paired bootstrap with p = 2 × min(P(Δ>0), P(Δ<0)),
consistent with the reported CIs.

Two PCC protocols are reported in the paper:

Table 8 (ablation) — per-channel means computed on the test batch (test-means).

Table 12 (headline PCC) — per-channel means estimated on the training split
and applied to the test split (train-means → test). This is the more conservative
protocol and the one reported as headline.

Known limitations
Documented in the paper's Section 6; repeated here for convenience.

Use/mention confound. Email and SMS carry use labels (the message
itself is fraudulent); forum carries mention labels (the post discusses
fraud). The model is trained jointly on all four roles. Leave-forum-out
partially controls for this, but the confound is not fully removed.

LLM-generated email corpus. E-PhishLLM is LLM-generated; the channel
shortcut may be partly a source artefact.

LLM annotation. 950 documents annotated zero-shot with Gemini,
validated on a 10% sample. No human inter-annotator agreement.

Residual cross-channel weakness. LOO-SMS accuracy for CIC-Intents is
0.243. PCC helps but does not solve transfer.

Small-n forum. Three independent estimates reported; hold-out (n = 17)
systematically underestimates forum performance relative to CV and LOO.

Single-seed main runs. Multi-seed verification on a sub-experiment
only (σ ≈ 0.01 macro-F1); forum CV variance σ ≈ 0.06.

Citation
bibtex
@article{bilikhodze2026channelshortcuts,
  title  = {Channel Shortcuts in Italian Multi-Channel Fraud Intent Detection},
  author = {Bilikhodze, Mariia},
  year   = {2026},
  note   = {University of Pavia, Department of Economics and Management}
}
License
Code (src/, notebooks/) — MIT (see LICENSE).

Data artefacts (data/splits/, results/*.json, results/*.csv) —
CC-BY-4.0 (see data/LICENSE).

Paper source (paper/main.tex, paper/main.pdf, paper/figures/) —
CC-BY-4.0 (see paper/LICENSE).

Original raw text retains the license of each source corpus — see
docs/data_sources.md.
