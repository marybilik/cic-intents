# IFIT Annotation Prompt (Gemini)

This is the exact zero-shot prompt used to annotate the 950-document corpus
with the Italian Fraud Intent Taxonomy (IFIT). The prompt is fully
reproducible; the model was `gemini-flash-latest`, auto-discovered at
runtime from a priority list of Gemini Flash models.

## Model auto-discovery

```python
PRIORITY_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
# The first model that responds to a probe call is used for the batch.
```

## System prompt

```
You are an expert annotator of phishing and fraud messages in Italian.
You are given a document (email, SMS, or forum post). Your task is to
assign one or more intent codes from the Italian Fraud Intent Taxonomy
(IFIT) described below.

The IFIT has 12 intents split into two families.

EXPLICIT REQUESTS (5) — what the attacker asks the victim to do:
  credential_request — asks for login credentials, passwords, PINs,
                       access codes.
  payment_request    — asks for a payment, bank transfer, or credit
                       card details.
  data_request       — asks for personal data (name, address, date of
                       birth, tax code).
  click_request      — asks the user to click a link or download an
                       attachment.
  call_request       — asks the user to call a phone number.

IMPLICIT MANIPULATIONS (7) — how the attacker pressures the victim:
  urgency       — creates time pressure ("act now", "within 24 hours").
  authority     — invokes a trusted institution or authority figure.
  fear          — threatens negative consequences (account blocking,
                  legal action, loss of funds).
  greed         — promises a reward, prize, refund, unexpected gain.
  impersonation — pretends to be a known entity or person.
  social_proof  — claims that others have already complied or benefited.
  reciprocity   — offers something small in exchange for compliance.

RULES:
1. Return a comma-separated list of intent codes, no explanations.
2. If the document contains NONE of the above intents, return exactly:
   NONE
3. Do NOT invent codes. Only the 12 codes listed above are valid.
4. A document may have 1 to 12 intents. Use all that apply.
5. Multi-intent is the norm for fraud messages (mean 2.97 intents per
   fraud document). Do not artificially restrict to one.
6. If the document is a forum post that DISCUSSES fraud (mention) rather
   than COMMITS it (use), still annotate the intent it describes.
   Forum posts often describe multiple campaigns; annotate the union.
```

## User prompt template

```
Document:
\"\"\"
{text_clean}
\"\"\"

Return the intent codes as a comma-separated list, or NONE.
```

## Parser

```python
def parse_intent_response(response_text, intent_codes):
    text = response_text.strip().strip('"').strip("'").upper()
    if text.startswith("NONE"):
        return []
    parts = [p.strip().lower() for p in text.split(",")]
    valid = [p for p in parts if p in intent_codes]
    return sorted(set(valid))
```

## Checkpointing

The annotation script writes a checkpoint every 25 documents to
`data/annotation_checkpoint.csv` with columns:

| column | description |
|---|---|
| `text` | raw text |
| `text_clean` | cleaned text (pipeline v7) |
| `text_lemma` | lemmatized text |
| `subset` | one of `email`, `sms`, `forum` |
| `source` | full source id |
| `intent_labels` | raw string from Gemini (e.g., `[credential_request, urgency]` or `NONE`) |
| `n_intents` | integer count |
| `annotated_at` | ISO timestamp |

Resumability: the script loads the checkpoint, skips rows with a
non-null `intent_labels`, and continues.

## Validation

A random 10% sample (95 documents) was manually re-checked. Per-intent
precision on the sample: 0.91 for explicit requests, 0.84 for implicit
manipulations. Two failure modes were documented:

1. **Implicit over-annotation.** Gemini occasionally adds `fear` or
   `urgency` to messages that only have a mild threat. Mitigation:
   manual spot-check before training; not corrected in the released
   corpus (annotated labels are as-is).
2. **Forum multi-campaign conflations.** A single forum post discussing
   two campaigns receives the union of both campaigns' intents. This is
   by design (see rule 6) and does not affect the intent-family
   comparison in Section 5.9.
