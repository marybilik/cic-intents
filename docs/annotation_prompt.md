# IFIT Annotation Prompt

This is the exact zero-shot prompt used to annotate 950 documents with
the Italian Fraud Intent Taxonomy (IFIT). The prompt was sent to
`openai/gpt-oss-120b` via the Groq API.

## System prompt
You are an expert annotator for Italian fraud detection research.
You will be given a short Italian text (email, SMS, or forum post).
Your task is to identify which manipulation intents are present.

The intent taxonomy has 12 categories, split into two families:

EXPLICIT (what the attacker asks for):

credential_request : asks for passwords, PINs, OTP, SPID, access codes

payment_request : asks for a payment, bank transfer, credit card

data_request : asks for personal data (name, address, tax code)

click_request : asks the user to click a link or download an attachment

call_request : asks the user to call a phone number

IMPLICIT (how the attacker manipulates):

urgency : time pressure ("act now", "within 24 hours")

authority : invokes a trusted institution

fear : threatens negative consequences

greed : promises a reward, prize, refund

impersonation : pretends to be a known entity or person

social_proof : claims others have already complied

reciprocity : offers something small in exchange for compliance

Rules:

Return ONLY a comma-separated list of intent codes.

If no intent applies, return the single token: NONE

Do not explain. Do not add quotes. Do not add any other text.

A document may have 0, 1, or many intents.

Prioritise precision over recall: only assign an intent if the
evidence in the text is explicit.


## User prompt (per document)

Text (channel: {channel}):
{text_clean}

Intents:

text

Where `{channel}` is one of `email`, `sms`, `telegram`, `forum`, and
`{text_clean}` is the preprocessed text (see `data_sources.md`).

## Parsing

The raw model output is parsed with the following logic:

```python
def parse_intents(raw_answer: str) -> list[str]:
    """Return a list of valid IFIT codes, or [] if NONE."""
    raw = raw_answer.strip().upper()
    if raw == "NONE" or raw == "" or raw.startswith("NONE"):
        return []
    # split on commas and remove any explanatory text
    parts = [p.strip() for p in raw.split(",")]
    valid = []
    for p in parts:
        # keep only tokens that match IFIT codes
        if p in IFIT_CODES:
            valid.append(p.lower())
        else:
            # try to match against code prefixes
            for code in IFIT_CODES:
                if code.upper() in p:
                    if code not in valid:
                        valid.append(code.lower())
                    break
    return valid
Validation
10% random sample manually verified.

Inter-annotator agreement not measured (single annotator).

Checkpoint written to disk every 25 documents — safe to resume on
interruption.

Full annotation log preserved with raw_answer, annot_id, and
is_done columns for full traceability.

Reproducing
python
from groq import Groq
import pandas as pd

client = Groq(api_key="...")  # set GROQ_API_KEY in env

def annotate(text: str, channel: str) -> list[str]:
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Text (channel: {channel}):\n{text}\n\nIntents:"},
        ],
        temperature=0.0,
    )
    return parse_intents(resp.choices[0].message.content)

df = pd.read_csv("data/corpus.csv")
for i, row in df.iterrows():
    if row["is_done"]:
        continue
    labels = annotate(row["text_clean"], row["source"])
    df.at[i, "intent_labels"] = ",".join(labels)
    df.at[i, "is_done"] = True
    if i % 25 == 0:
        df.to_csv("data/corpus_annotated.csv", index=False)
Cost note
At the time of the experiments (2025), the Groq API pricing for
gpt-oss-120b was ~$0.15 per 1M input tokens. The full 950-document
annotation cost less than $1.

