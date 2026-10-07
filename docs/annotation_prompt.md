# IFIT Annotation Prompt

This is the exact zero-shot prompt used to annotate 950 documents with
the Italian Fraud Intent Taxonomy (IFIT). Annotation was performed via
the **Google Gemini API** (`gemini-flash-latest`, with runtime fallback
to alternative models if quota is exhausted).

## Model selection

At runtime the notebook auto-discovers the first working model from
this list:

| Priority | Model name |
|---|---|
| 1 | `gemini-flash-latest` |
| 2 | `gemini-2.5-flash` |
| 3 | `gemini-2.5-flash-lite` |
| 4 | `gemini-flash-lite-latest` |
| 5 | `gemini-2.0-flash-001` |
| 6 | `gemini-2.0-flash-lite` |

Both `v1beta` and `v1` API versions are tried. The first model that
returns HTTP 200 is used for the entire annotation run.

## Generation config

```python
{
    "temperature": 0.0,
    "maxOutputTokens": 200,
}
Input text is truncated to 2000 characters (text[:2000]) before
being sent.

Prompt (Italian, as used in the paper)
The prompt is formulated in Italian and includes the full IFIT taxonomy
inline as a bulleted list, followed by four strict rules.

text
Analizza il testo e identifica TUTTI gli intenti fraudolenti.

CODICI:
- credential_request: Richiesta login/password/PIN/SPID
- payment_request: Richiesta pagamento/bonifico/dati carta/IBAN
- data_request: Richiesta codice fiscale/documento/indirizzo
- click_request: Richiesta click su link/allegato/QR code
- call_request: Richiesta chiamare un numero/contattare falso supporto
- urgency: Pressione temporale, scadenza
- authority: Riferimento a banca/Poste/INPS/polizia/ministero
- fear: Minaccia: blocco/multa/sanzione/azione legale
- greed: Promessa premio/bonus/rimborso/regalo
- impersonation: Fingere di essere conoscente/collega/parente
- social_proof: Riferimento ad altri utenti
- reciprocity: Senso di debito ("abbiamo già fatto per te")

REGOLE:
1. Multi-label (uno o più codici)
2. Rispondi SOLO con i codici separati da virgola
3. Se non ci sono intenti: NONE
4. Nessuna spiegazione

Testo:
\"\"\"
{text}
\"\"\"

Codici:
Where {text} is the preprocessed text_clean field (see
data_sources.md for the preprocessing pipeline).

Parsing
The model output is parsed with a regex that matches IFIT codes as
word-boundary tokens:

python
import re

ITALIAN_FRAUD_INTENTS = {
    'credential_request': 'Richiesta login/password/PIN/SPID',
    'payment_request':    'Richiesta pagamento/bonifico/dati carta/IBAN',
    'data_request':       'Richiesta codice fiscale/documento/indirizzo',
    'click_request':      'Richiesta click su link/allegato/QR code',
    'call_request':       'Richiesta chiamare un numero/contattare falso supporto',
    'urgency':            'Pressione temporale, scadenza',
    'authority':          'Riferimento a banca/Poste/INPS/polizia/ministero',
    'fear':               'Minaccia: blocco/multa/sanzione/azione legale',
    'greed':              'Promessa premio/bonus/rimborso/regalo',
    'impersonation':      'Fingere di essere conoscente/collega/parente',
    'social_proof':       'Riferimento ad altri utenti',
    'reciprocity':        'Senso di debito ("abbiamo già fatto per te")',
}
INTENT_CODES = list(ITALIAN_FRAUD_INTENTS.keys())

def parse_intents(answer: str) -> list[str]:
    if not answer:
        return []
    if re.search(r'\bNONE\b', answer, re.IGNORECASE):
        return []
    pattern = '|'.join(re.escape(c) for c in INTENT_CODES)
    found = re.findall(pattern, answer.lower())
    seen, unique = set(), []
    for c in found:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
Rate limiting
Gemini's free tier allows 15 requests per minute. The annotation
loop enforces a 4.5-second sleep between requests (~13 RPM, with
headroom), plus exponential backoff on HTTP 429:

python
if r.status_code == 429:
    time.sleep(5 * (2 ** attempt))  # 5, 10, 20 s
Checkpointing and resumption
A checkpoint is written to disk every 25 documents.

Two copies are kept: a working copy in the Colab filesystem and a
backup in Google Drive (/content/drive/MyDrive/fraud_project/).

Interrupted runs are safely resumable: the notebook re-loads the
checkpoint and skips rows where intent_labels is already a valid
list representation (i.e., '[...]').

Rows with raw_answer matching failed or error are automatically
reset and re-annotated on the next run.

Validation
10% random sample manually verified.

Inter-annotator agreement was not measured (single annotator, single
model).

Full annotation log is preserved in annotation_checkpoint.csv with
text_clean, intent_labels, raw_answer, subset, and is_done
columns for full traceability.

Reproducing the annotation
python
import os, re, time, requests
import pandas as pd

GEMINI_KEY = os.environ["GEMINI_API_KEY"]

CANDIDATES = [
    'gemini-flash-latest', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
    'gemini-flash-lite-latest', 'gemini-2.0-flash-001', 'gemini-2.0-flash-lite',
]

def discover_model():
    for api_ver in ('v1beta', 'v1'):
        for model_name in CANDIDATES:
            url = (f"https://generativelanguage.googleapis.com/{api_ver}/models/"
                   f"{model_name}:generateContent?key={GEMINI_KEY}")
            try:
                r = requests.post(url, json={
                    "contents": [{"parts": [{"text": "Reply only with: OK"}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10}
                }, timeout=15)
                if r.status_code == 200:
                    return model_name, api_ver
            except Exception:
                pass
    raise RuntimeError("No working Gemini model found.")

MODEL_NAME, API_VER = discover_model()
URL = (f"https://generativelanguage.googleapis.com/{API_VER}/models/"
       f"{MODEL_NAME}:generateContent?key={GEMINI_KEY}")

def annotate(text: str) -> tuple[list[str], str]:
    if not isinstance(text, str) or len(text) < 30:
        return [], 'skip'
    payload = {
        "contents": [{"parts": [{"text": PROMPT.format(text=text[:2000])}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
    }
    r = requests.post(URL, json=payload, timeout=30)
    r.raise_for_status()
    raw = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    return parse_intents(raw), raw
Cost note
At the time of the experiments (2025–2026), Google's Gemini free tier
was used for all 950 documents. Total cost: $0.
