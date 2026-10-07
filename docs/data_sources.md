# Data sources and provenance

The CIC-Intents corpus is assembled from four independently collected
Italian-language sources. We do **not** redistribute raw text — only the
processed splits, test-set predictions, and derived metrics. Each source
is described below with its license and download instructions.

## Summary

| Channel | Source | n (total) | n (fraud) | n (legit) | Role | Type |
|---|---|---|---|---|---|---|
| Email | E-PhishGen | 2,701 | 1,131 | 1,570 | use | LLM-generated |
| SMS | Smishing-IMC25 | 536 | 536 | 0 | use | real, user reports |
| Telegram | MuLTa-Telegram | 827 | 0 | 827 | mention | non-hateful subset |
| Forum | Scraped (Digital-Forum, MilanWorld) | 312 | 0 | 312 | mention | annotated by us |
| **Total** | | **4,376** | **1,667** | **2,709** | | |

## Email — E-PhishGen

- **Paper:** Pajola, Pogliani, Balzarotti (2025), *E-PhishGen: Unlocking
  Novel Research in Phishing Email Detection*, AISec '25.
- **HuggingFace:** https://huggingface.co/datasets/pajola/e-phishGen
- **License:** see HF dataset card
- **Language split:** the full corpus has 16,616 emails in English,
  Italian, and German. We use the Italian subset (16.3% = 2,701 emails).
- **Role:** `use` — every fraud email *is* the attack.

**Download:**

```python
from datasets import load_dataset
ds = load_dataset("pajola/e-phishGen", split="train")
italian = ds.filter(lambda x: x["language"] == "it")
italian.to_csv("data/raw/email_italian.csv", index=False)
SMS — Smishing-IMC25
Paper: Agarwal et al. (2025), presented at ACM Internet
Measurement Conference (IMC) 2025.

GitHub: https://github.com/reportsmishing/Smishing-Dataset-IMC25

License: see repository

Size: over 64,000 raw user reports pre-deduplication; 536 unique
fraudulent SMS after deduplication.

Average length: 115 characters.

Role: use — every SMS is the attack.

Download:

bash
git clone https://github.com/reportsmishing/Smishing-Dataset-IMC25.git
cp Smishing-Dataset-IMC25/*.csv data/raw/sms_raw.csv
Telegram — MuLTa-Telegram
Paper: Leonardelli et al. (2025), MuLTa-Telegram: A Multi-Label
Hate Speech Dataset for Italian and Polish.

GitHub: https://github.com/dhfbk/MuLTa-Telegram

License: see repository

Size: ~4,000 Telegram messages in Italian and Polish. Italian split
has 2,002 messages, of which 20.5% are hateful. We use the non-hateful
Italian subset (827 messages).

Role: mention — the messages discuss topics, they don't commit
fraud. Used as a proxy for legitimate conversational Italian.

Caveat: a dedicated legitimate Italian Telegram corpus for fraud
detection does not exist publicly. We document this as a limitation of
the study and treat Telegram as a proxy for conversational register.

Forum — scraped from two Italian forums
Sources: Digital-Forum (https://www.digital-forum.it) and
MilanWorld (https://www.milanworld.net).

Scraper: cloudscraper (Python) to bypass Cloudflare.

Size: 312 posts discussing phishing and smishing campaigns.

Role: mention — the posts talk about fraud.

Ethical note: we scrape only public posts, do not redistribute
usernames or personal information, and use the data exclusively for
defensive research. Post-processing strips all identifiers before they
appear in the corpus.

Download: we cannot redistribute the raw forum posts. Instead we
provide the annotated text_clean field for the 312 posts in
data/splits/forum_processed.csv.

Preprocessing pipeline
Applied identically across all four channels, in this order:

Strict HTML stripping — whitelist of genuine tags only
(<p>, <br>, <a>, <b>, <i>); everything else removed.

Entity-tag normalization — <NAMED_ENTITY> → [ENTITY],
<URL> → [URL], <EMAIL> → [EMAIL], <PHONE> → [PHONE].

HTML entity decoding — &amp; → &, &nbsp; → space, etc.

URL / email / phone placeholder substitution — regex-based.

Homoglyph normalization — Cyrillic-to-Latin, applied only when
fewer than 5% of characters are Cyrillic (preserves genuine
mixed-language content).

Lemmatization — spaCy it_core_news_sm.

Marker extraction — four psycholinguistic markers:

urgency, authority, fear, greed

plus two cross-products (urgency_x_length, etc.).

Retention rate: 4,376 / 4,578 = 95.6% of the raw pool.

Channel divergence
Email fraud averages 1.98 urgency markers per document; SMS fraud
averages 0.18 — an 11× gap. Italian SMS attackers rely on
impersonation of a trusted institution (authority) and on threat of
account blocking (fear), not on time pressure. This contradicts the
usual characterization of phishing as urgency-driven and directly
motivates the multi-channel design of IFIT.

License
All redistributed artefacts (splits, predictions, metrics) are released
under CC-BY-4.0 — see data/LICENSE.
The original raw text is governed by the terms of each source dataset.

text

`Commit changes...` → `Commit changes`.

---

## Файл 10/13: `docs/reproducibility.md`

`Add file` → `Create new file` → имя `docs/reproducibility.md` → вставить:

```markdown
# Reproducibility checklist

This document lists everything a reviewer needs to reproduce the results
in the paper. It follows the format recommended by the
[NeurIPS 2024 reproducibility checklist](https://neurips.cc/public/guides/PaperChecklist)
and the Elsevier "Artifacts" requirements.

## 1. Code

- **Repository:** https://github.com/marybilik/cic-intents
- **Entry point:** `notebooks/cic_intents_full_pipeline.ipynb`
- **Notebook is self-contained.** It downloads/loads data, trains models,
  evaluates, and generates all figures.
- **Environment:** Python 3.10+; dependencies in `requirements.txt`.
- **Hardware used:** NVIDIA Tesla T4 (15.6 GB) via Kaggle Notebooks;
  post-hoc analyses on CPU.

## 2. Randomness

- **Seed:** 42
- **Where set:**
  ```python
  import random, numpy as np, torch
  random.seed(42); np.random.seed(42)
  torch.manual_seed(42); torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
Multi-seed verification: a sub-experiment with 3 seeds produced
σ ≈ 0.01 in macro-F1, within the reported confidence intervals.
Full multi-seed runs are not reported in the paper (single-seed for
all headline numbers).

3. Training hyperparameters
Parameter	Value
Encoder	Musixmatch/umberto-commoncrawl-cased-v1
Optimizer	AdamW
Learning rate	5 × 10⁻⁵
Weight decay	0.01
Batch size	32
Max sequence length	160
Epochs	10
τ (SupCon temperature)	0.07
λ (SupCon weight)	0.3
λ (MMD weight)	0.3
α (channel weight)	0.3
pos_weight clipping	5
MMD kernel bandwidths	{1, 2, 4, 8}
4. Data splits
Split	Construction	N
Main train	70% stratified by channel + intent	665
Main val	15% stratified	142
Main test	15% stratified	143
LOO-forum train	email + SMS only (all)	711
LOO-forum val	email + SMS only (all)	126
LOO-forum test	all forum	113
Forum 5-fold CV	80/20 per fold, stratified	113
Stratification key: (channel_id, has_any_intent).

5. Evaluation protocols
Main split
Single hold-out of 143 documents, stratified as above. Metrics computed
with threshold 0.5 for F1, and via full ranking for mAP / Top-k.

Forum small-n
Three independent estimates, all reported:

Hold-out (n = 17): the naive 70/15/15 contribution.

5-fold stratified CV (n = 113): mean ± std across folds.

Leave-forum-out (n = 113): model trained on email+SMS only,
tested on all 113 forum documents.

The 4 pp gap between LOO (0.220) and CV (0.255) quantifies how much
forum-specific training helps.

Bootstrap CIs
1000 resamples; 2.5th and 97.5th percentiles. Important: for mAP,
the intent set is fixed across resamples (the same nine intents with
support ≥ 2 in the original test set). Otherwise, the two intents with
support = 1 would be excluded on many resamples, inflating the mean.

Paired tests
Paired bootstrap test with 1000 resamples, paired by example index.

6. Metric definitions
mAP (macro): mean of Average Precision over the nine intents with
test-set support ≥ 2.

Macro-F1: unweighted mean of per-intent F1 at threshold 0.5.

Top-k recall: fraction of ground-truth intents appearing in the
top-k predicted intents, averaged over documents with ≥ 1 true intent.

Domain MI: normalized mutual information between channel labels
and predictions of a 5-fold cross-validated logistic-regression probe
on the CLS embeddings.

PN (Probability of Necessity): 
f
t
(
x
)
−
f
t
(
x
∖
T
k
(
x
)
)
f 
t
​
 (x)−f 
t
​
 (x∖T 
k
​
 (x)).

PS (Probability of Sufficiency): 
f
t
(
T
k
(
x
)
)
−
f
t
(
x
∖
T
k
(
x
)
)
f 
t
​
 (T 
k
​
 (x))−f 
t
​
 (x∖T 
k
​
 (x)).

7. Ablation study
Five configurations on the main split:

Config	λ_SupCon	λ_MMD	Post-hoc centering
BCE only	0	0	no
+ SupCon	0.3	0	no
+ SupCon + MMD	0.3	0.3	no
+ SupCon + MMD + Centering	0.3	0.3	yes
Centering only	0	0	yes
Each trained with the same hyperparameters and split. +SupCon+MMD
reuses cicl_main.pt; the other four are separately trained.

8. Figures
All 11 figures in paper/figures/ are generated by
notebooks/cic_intents_full_pipeline.ipynb, cells numbered
# CELL 13 and following. Figure 1's training curves use the LOO-forum
variant (hist_loo), which is representative of all runs; the main
model's history file was not retained during export, but its convergence
behaviour matches the plotted run within 1–2 pp (documented in the
figure caption).

9. Checkpoints
Trained checkpoints are hosted externally (Kaggle dataset checks):

File	Size	Description
cicl_main.pt	445 MB	Main model (email + SMS + forum)
cicl_loo.pt	445 MB	LOO-forum model (email + SMS only)
baseline_xlmr.pt	~1.1 GB	XLM-R-base with CIC-Intents framework
baseline_mdeberta.pt	~1.1 GB	mDeBERTa-v3-base with CIC-Intents framework
abl_bce_only.pt	445 MB	BCE-only ablation
abl_supcon.pt	445 MB	SupCon-only ablation
To use: download from the Kaggle dataset, place in models/, and
update the paths in the notebook. See models/README.md.

10. Reported numbers
All numbers in paper/main.tex are traced to
results/paper_metrics.json and the CSV files in results/. If any
number differs between the paper and the code, the code is authoritative.

11. How to reproduce end-to-end
bash
git clone https://github.com/marybilik/cic-intents.git
cd cic-intents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download data (see docs/data_sources.md)
mkdir -p data/raw && python scripts/download_data.py

# download checkpoints (see models/README.md)
mkdir -p models && python scripts/download_checkpoints.py

# run the pipeline
jupyter lab notebooks/cic_intents_full_pipeline.ipynb
Total training time on T4: ~4 hours for all models + post-hoc analyses.

12. Known issues
hist_main not retained: the main model's training history was
not saved during the original run. Figure 1 shows the LOO-forum
variant instead. We document this in the figure caption.

Forum small-n: the naive 70/15/15 split gives only 17 test
documents for the forum channel, which produces unstable estimates.
We mitigate this with 5-fold CV and leave-forum-out. All three
numbers are reported.

Twitter/Telegram proxy: no legitimate Italian fraud corpus for
Telegram exists; we use a non-hateful subset of MuLTa-Telegram as a
proxy and document this as a limitation.
