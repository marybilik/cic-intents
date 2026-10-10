# End-to-End Pipeline

Full walkthrough from raw sources to paper figures. Two parts, matching
the two notebooks.

## Part A — Data acquisition & preprocessing (Colab, internet required)

Notebook: [`../notebooks/01-corpus-construction.ipynb`](../notebooks/01-corpus-construction.ipynb).

### A1. Forum scraping

```python
import cloudscraper, time, random
from bs4 import BeautifulSoup
# Full scraper in notebook 01, CELL 4.
```

Output: `forum_corpus_max.csv` — 312 posts after length filter.

### A2. Load HuggingFace corpora

```python
from huggingface_hub import hf_hub_download
import json, pandas as pd

# E-PhishLLM
fp = hf_hub_download(repo_id="pajola/e-phishGen",
                     filename="ephishLLM.json", repo_type="dataset")
data = json.load(open(fp))
df_e = pd.DataFrame(data)
df_e = df_e[df_e['Language'] == 'it'].copy()
df_e['text'] = df_e['Subject'].fillna('') + ' ' + df_e['Body'].fillna('')
df_e = df_e[['text', 'type']].rename(columns={'type': 'target'})
df_e['source'] = 'email_ephish_it'

# Smishing-IMC25  (git clone)
# git clone https://github.com/reportsmishing/Smishing-Dataset-IMC25.git
df_s = pd.read_csv('Smishing-Dataset-IMC25/dataset/final_dataset_output.csv')
df_s = df_s[df_s['language'].str.lower().str.contains('italian|^it$', na=False)]
df_s = df_s[['text']].copy()
df_s['target'] = 1
df_s['source'] = 'sms_imc25_fraud'

# MuLTa-Telegram  (git clone)
# git clone https://github.com/dhfbk/MuLTa-Telegram.git
multa = json.load(open('MuLTa-Telegram/data/it/train.json'))
df_m = pd.DataFrame(multa)
df_m = df_m[df_m['hate label'] == 0][['text']].copy()
df_m['target'] = 0
df_m['source'] = 'multa_telegram'
```

### A3. Cleaning pipeline v7

See `src/cleaning.py::clean_text`. Applied to all four channels.

Order: strict HTML → entity tags → HTML entities → URL/email/phone →
Cyrillic homoglyph handling → zero-width + whitespace + lowercase.

### A4. Deduplication & filtering

1. Drop texts shorter than 30 chars after cleaning.
2. Within-source dedup on `text_clean` (keep first).
3. Cross-source dedup with priority email > SMS > Telegram.
4. Exclude forum texts that appear in the labeled corpus.

Result: 4,064 labeled + 312 forum = **4,376 documents** (95.6%
retention).

### A5. Lemmatization

```python
import spacy
nlp = spacy.load('it_core_news_sm')

def lemmatize_batch(texts, batch_size=256):
    docs = nlp.pipe(texts.astype(str), batch_size=batch_size,
                    disable=['ner', 'parser'])
    return [' '.join(t.lemma_ for t in doc
                     if not t.is_stop and not t.is_punct
                     and not t.is_space and len(t.text) > 2)
            for doc in docs]
```

### A6. Psycholinguistic markers

Four markers (`urgency`, `authority`, `fear`, `greed`), word-list based.
See `MARKERS` in notebook 01, CELL 7. Used in the soft-gate baseline.

### A7. IFIT annotation (Gemini)

See [`annotation_prompt.md`](annotation_prompt.md). Output:
`df_cicl.csv` — 950 docs, `multihot` column as a stringified NumPy
array of length 12.

---

## Part B — Training & evaluation (Kaggle / local GPU)

Notebook: [`../notebooks/02-training-evaluation.ipynb`](../notebooks/02-training-evaluation.ipynb).

### B1. Load annotated corpus

```python
import pandas as pd, numpy as np
df = pd.read_csv('data/df_cicl.csv')
df['multihot'] = df['multihot'].apply(
    lambda s: np.array(eval(s), dtype=np.float32))
df['channel_id'] = df['channel_id'].astype(int)
df['n_intents']  = df['multihot'].apply(lambda v: int(v.sum()))
```

### B2. Build splits

```python
from src.splits import make_splits
splits = make_splits(df)
# splits['main']['test']  -- 143 docs
# splits['loo']['test']   -- all 113 forum
# splits['cv']['folds']   -- 5 folds
```

### B3. Train

```python
from transformers import AutoTokenizer
import torch
from src.train import train_cicl

tokenizer = AutoTokenizer.from_pretrained(
    'Musixmatch/umberto-commoncrawl-cased-v1')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, history = train_cicl(
    splits['main']['train'],
    splits['main']['val'],
    tokenizer=tokenizer,
    model_name='Musixmatch/umberto-commoncrawl-cased-v1',
    n_intents=12,
    device=device,
    epochs=10, lr=5e-5,
    lambda_supcon=0.3, lambda_mmd=0.3)

torch.save(model.state_dict(), 'models/cicl_main.pt')
```

### B4. Evaluate

```python
from src.utils import predict_proba
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support)

probs, ys, chs = predict_proba(model, splits['main']['test'],
                                tokenizer, device)
preds = (probs > 0.5).astype(int)

_, _, f1, _ = precision_recall_fscore_support(
    ys, preds, average='macro', zero_division=0)

# mAP over intents with support >= 2 (paper protocol)
supports = ys.sum(axis=0).astype(int)
aps = [average_precision_score(ys[:, i], probs[:, i])
       for i in range(12) if supports[i] >= 2]
mAP = float(np.mean(aps))

# Top-5 recall
hits, total = 0, 0
for row in range(len(probs)):
    top5 = np.argsort(-probs[row])[:5]
    tset = set(np.where(ys[row] > 0.5)[0])
    if not tset:
        continue
    hits += len(tset & set(top5)); total += len(tset)
top5_recall = hits / total

print(f"mAP={mAP:.4f} | Top-5={top5_recall:.4f} | F1={f1:.4f}")
# Expected: mAP=0.7210 | Top-5=0.9310 | F1=0.4654
```

### B5. Post-hoc Channel Centering (PCC)

```python
from src.pcc import evaluate_pcc
results = evaluate_pcc(model, splits['main']['test'], tokenizer, device)
print(f"MI: {results['mi_raw']:.3f} → {results['mi_centered']:.3f} "
      f"({results['reduction']*100:.1f}%)")
# Expected (train-means → test protocol):
#   MI: 0.676 → 0.116 (−82.9%)
```

> **Note on the two PCC protocols.**
> The headline number uses per-channel means estimated on the training
> split and applied to the test split (train-means → test). The ablation
> study (Table 8 in the paper) uses a different protocol in which
> per-channel means are computed on the test batch itself (test-means),
> giving 0.676 → 0.442 for the full model and 0.788 → 0.380 for the
> BCE-only model. Both protocols are documented in the paper.

### B6. Causal attribution (PN/PS)

```bash
python -m src.attribution --checkpoint models/cicl_main.pt \
                          --data data/df_cicl.csv \
                          --out results/pnps.csv
```

Expected: PN = 0.207 / 0.294 / 0.320 at k = 5, 10, 20.

### B7. Figures

```bash
python -m src.figures --checkpoint models/cicl_main.pt \
                      --data data/df_cicl.csv \
                      --outdir results/figures/
```

---

## Reproducibility checklist

| Item | Value |
|---|---|
| Random seed | 42 |
| Encoder | UmBERTo-CommonCrawl-Cased-v1 |
| Optimizer | AdamW, lr 5e-5, wd 0.01 |
| Batch | 32 |
| Max seq len | 160 |
| Epochs | 10 |
| τ (SupCon temperature) | 0.07 |
| α (channel weight) | **0.5** |
| λ_SupCon | 0.3 |
| λ_MMD | 0.3 |
| pos_weight clip | 5 |
| Bootstrap | 1000 resamples |
