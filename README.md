# CIC-Intents

Channel-Invariant, Interpretable and Causally Grounded Intent Detection
for Italian Multi-Channel Fraud.

Code, prompts, data splits and reproduction scripts for the paper
(M. Bilikhodze, University of Pavia).

---

## Repository structure
.
├── data/ # LICENSE, README (data not shipped, see data/README.md)
├── docs/ # paper-level documentation
│ ├── annotation_prompt.md # exact Gemini prompt + parser
│ ├── data_sources.md # provenance of all four channels
│ ├── pipeline.md # end-to-end description
│ ├── reproducibility.md # step-by-step checklist
│ └── taxonomy.md # Italian Fraud Intent Taxonomy (IFIT)
├── models/ # checkpoints (hosted externally, see models/README.md)
├── notebooks/ # lightweight demo + reproduction notebooks
├── paper/ # LaTeX source
├── results/ # metrics, predictions, CSVs (see results/README.md)
├── src/ # modular pipeline
│ ├── attribution.py # PN/PS gradient×input saliency
│ ├── cleaning.py # preprocessing v7 (HTML, homoglyphs, spaCy lemmas)
│ ├── features_v4.py # 33 hand-crafted intent features
│ ├── figures.py # all paper figures
│ ├── losses.py # MultiLabelSupConLoss + channel_mmd_loss
│ ├── model.py # CICLMMDv2 (UmBERTo + intent head + projection + MMD)
│ ├── pcc.py # Post-hoc Channel Centering + domain MI
│ ├── splits.py # 70/15/15 + LOO-forum + 5-fold CV
│ ├── train.py # train_cicl + predict_proba + get_cls_embeddings
│ └── utils.py # seeds, paths, IFIT codes, parsing
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt

text

---

## Install

```bash
git clone https://github.com/marybilik/cic-intents
cd cic-intents
pip install -r requirements.txt
python -m spacy download it_core_news_sm
Requires a GPU with ≥ 15 GB VRAM for training (T4 / P100).
Post-hoc analyses (bootstrap, causal attribution, figures) run on CPU.

Data
The four-channel Italian corpus is not shipped in this repository.
It is assembled from four independently collected sources and, for two of
them, from a scraped corpus that requires network access. See
docs/data_sources.md for provenance and licensing.

The final annotated corpus is exported as a single file df_cicl.csv
(950 rows, 12-intent multi-hot labels). It is available on request and as
part of the Kaggle dataset mariabilik/df-cicl.

To rebuild the corpus from scratch:

bash
export GEMINI_API_KEY="..."          # annotation
export CIC_BUILD_CORPUS=1            # re-scrape + re-annotate
python -m src.cleaning               # preprocessing v7
python -m src.attribution --annotate # Gemini annotation
The exact annotation prompt is in docs/annotation_prompt.md.

Training & evaluation
Trained checkpoints are hosted externally — see models/README.md.
If the checkpoints are present at $CIC_CKPT_DIR (default ./models/),
training is skipped and evaluation runs directly from them.

Otherwise, to train from scratch:

bash
export CIC_DATA_DIR="./data"         # where df_cicl.csv lives
export CIC_OUT_DIR="./results"
python -m src.train --config configs/main.yaml
python -m src.pcc                    # Post-hoc Channel Centering
python -m src.attribution            # PN/PS causal attribution
python -m src.figures                # generate all paper figures
Environment variables:

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
Metric	Value
mAP (macro, 9 intents, support ≥ 2)	0.723
Mean AUC	0.784
Top-5 recall	0.924
Macro-F1 (t = 0.5)	0.464
Email macro-F1	0.452
SMS macro-F1	0.453
Domain MI: raw → centered	0.739 → 0.335 (−54.7%)
k-NN intent |Δ| after centering	0.009
PN at k = 5, 10, 20	0.183 / 0.286 / 0.356
Full per-intent and per-channel numbers are in results/paper_metrics.json
and in the paper.

Reproducibility
See docs/reproducibility.md for the full checklist. Summary:

Random seed 42 (numpy, torch, sklearn splits).

Frozen hyperparameters (see table above).

Splits: main 70/15/15 (N_test = 143), LOO-forum (n = 113), 5-fold CV on forum.

Bootstrap CIs: 1000 resamples, seed 42.

Paired tests: paired bootstrap with p = 2 × min(P(Δ>0), P(Δ<0)), consistent
with the reported CIs.

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

Residual cross-channel weakness. LOO-SMS accuracy is 0.243.
PCC helps but does not solve transfer.

Small-n forum. Three independent estimates reported.

Single-seed main runs. Multi-seed verification on a sub-experiment
only (σ ≈ 0.01 macro-F1); forum CV variance σ ≈ 0.06.

Citation
bibtex
@article{bilikhodze2026cicintents,
  title  = {CIC-Intents: Channel-Invariant, Interpretable and Causally
            Grounded Intent Detection for Italian Multi-Channel Fraud},
  author = {Bilikhodze, Mariia},
  year   = {2026},
  note   = {University of Pavia, Department of Economics and Management}
}
License
MIT (see LICENSE).
