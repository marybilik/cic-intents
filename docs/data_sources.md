# Data sources

Four independently collected Italian channels, **4,376 documents**
total. **950 documents** carry IFIT intent annotations (used for
training and evaluation).

## Email — E-PhishLLM

- **Source:** `pajola/e-phishGen` on HuggingFace.
- **Reference:** Pajola et al., 2025.
- **Total:** 2,701 Italian emails (1,131 fraudulent, 1,570 legitimate).
- **Role:** use (fraud) / legit.
- **Caveat:** LLM-generated. This is the only publicly available Italian
  phishing email corpus at scale. The LLM-generation is a limitation
  acknowledged in the paper (Section 6).

## SMS — Smishing-IMC25

- **Source:** `reportsmishing/Smishing-Dataset-IMC25` on GitHub.
- **Reference:** Agarwal et al., 2025 (ACM IMC 2025).
- **Total:** 536 fraudulent SMS after deduplication.
- **Average length:** 115 characters.
- **Role:** use.
- **Provenance:** mined from over 64,000 public user reports of
  smishing attacks across five Italian forums.

## Telegram — MuLTa

- **Source:** `dhfbk/MuLTa-Telegram` on GitHub.
- **Reference:** Leonardelli et al., 2025 (LREC 2026).
- **Total:** 827 non-hateful Italian Telegram messages
  (filtered from 2,002 Italian messages; 20.5% are hateful).
- **Role:** legit (proxy for non-fraud conversational messaging).
- **Caveat:** a legitimate Telegram fraud corpus is not publicly
  available. We use the non-hateful Italian subset as a proxy, with the
  explicit caveat that this is a controlled constraint.

## Forum — scraped

- **Sources:** Digital-Forum (2 threads), MilanWorld (1 thread).
- **Method:** `cloudscraper` + BeautifulSoup, XenForo parser.
- **Total:** 312 posts after length filtering (> 100 characters).
- **Role:** mention (the post discusses phishing/smishing campaigns).
- **Annotation:** 113 posts annotated with IFIT intents by us.

## Aggregation

After preprocessing (v7), length filtering (≥ 30 characters after
cleaning), and cross-source deduplication:

| Channel | Total | Fraud | Non-fraud | Role |
|---|---:|---:|---:|---|
| Email (E-PhishLLM) | 2,701 | 1,131 | 1,570 | use / legit |
| SMS (Smishing-IMC25) | 536 | 536 | 0 | use |
| Telegram (MuLTa) | 827 | 0 | 827 | mention |
| Forum | 312 | 0 | 312 | mention |
| **Total** | **4,376** | **1,667** | **2,709** | |

Annotation covers **950 documents** — email 421 (fraud + legit),
SMS 416 (fraud only), forum 113 (mention), Telegram 0
(Telegram is baseline-only, not annotated).

## Licensing

Each source retains its original license. E-PhishGen, Smishing-IMC25 and
MuLTa are public research datasets; the forum corpus was scraped from
public web pages. Redistribution of the aggregated corpus is subject to
the most restrictive upstream license. Contact the author for access.
