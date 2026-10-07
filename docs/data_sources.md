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
