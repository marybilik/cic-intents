# Annotation prompt

The corpus was annotated zero-shot with **Google Gemini** (`gemini-flash-latest`,
auto-discovered at runtime from a priority list of Gemini Flash models).
The same prompt is used in `src/attribution.py --annotate`.

## Model discovery

Before annotation, the script probes a priority list of Gemini models
across two API versions (`v1beta`, `v1`) and picks the first one that
responds with HTTP 200:

```python
CANDIDATES = [
    'gemini-flash-latest',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-flash-lite-latest',
    'gemini-2.0-flash-001',
    'gemini-2.0-flash-lite',
]
API_VERSIONS = ['v1beta', 'v1']
```

## Prompt template

```
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
- reciprocity: Senso di debito

REGOLE:
1. Multi-label (uno o più codici)
2. Rispondi SOLO con i codici separati da virgola
3. Se non ci sono intenti: NONE
4. Nessuna spiegazione

Testo:
"""
{text}
"""

Codici:
```

## Parser

```python
def parse_answer(ans, intent_codes):
    if re.search(r'\bNONE\b', ans, re.IGNORECASE):
        return []
    pattern = '|'.join(re.escape(c) for c in intent_codes)
    found = re.findall(pattern, ans.lower())
    seen, uniq = set(), []
    for c in found:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq
```

## Rate limiting

- Sleep 4.5 s between requests (~13 RPM, below the 15 RPM free-tier limit)
- Exponential backoff on HTTP 429: 5 s, 10 s, 20 s
- Checkpoint written every 25 documents
- Failed rows are reset and retried on the next run

## Validation

A random 10% sample of the annotated corpus was manually checked for
label correctness. No inter-annotator agreement study was performed;
this is acknowledged in the paper (Section 6).

## Reproducibility

To re-annotate from scratch:

```bash
export GEMINI_API_KEY="..."
python -m src.attribution --annotate --out ./data/annotation_full.csv
```

Output format: one row per document with columns
`text_clean`, `subset`, `source`, `intent_labels` (list), `raw_answer`.
