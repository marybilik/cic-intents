# The Italian Fraud Intent Taxonomy (IFIT)

12 intents, multi-label, in two families. Adapted from DIDECO
(Popovic et al., 2026) for Italian and narrowed to 12 categories to
reflect the observation that most Italian fraud messages target 2–3
intents simultaneously (mean **2.97** intents per positive document in
our annotated corpus).

## Explicit requests (5)

| Code | Definition | Lexical cues |
|---|---|---|
| `credential_request` | Asks for login credentials, passwords, PINs, access codes. | "accedi", "conferma le credenziali", "il tuo PIN", "verifica i tuoi dati di accesso" |
| `payment_request` | Asks for a payment, bank transfer, or credit card details. | "bonifico", "pagamento", "carta di credito", "IBAN", "importo" |
| `data_request` | Asks for personal data: name, address, date of birth, tax code. | "codice fiscale", "data di nascita", "indirizzo", "documento" |
| `click_request` | Asks the user to click a link or download an attachment. | "clicca qui", "segui il link", "apri l'allegato", "scarica il file" |
| `call_request` | Asks the user to call a phone number. | "chiama il", "contatta il numero", "richiama il" |

## Implicit manipulations (7)

| Code | Definition | Lexical cues |
|---|---|---|
| `urgency` | Creates time pressure. | "entro 24 ore", "subito", "scade oggi", "ultimo avviso" |
| `authority` | Invokes a trusted institution or authority figure. | "la tua banca", "Poste Italiane", "INPS", "Ministero", "Polizia" |
| `fear` | Threatens negative consequences. | "account bloccato", "sospensione", "azione legale", "perdita di fondi" |
| `greed` | Promises a reward, prize, refund, unexpected gain. | "bonus", "rimborso", "premio", "vincita", "investimento" |
| `impersonation` | Pretends to be a known entity or person. | references to specific banks, services, or individuals with false authority |
| `social_proof` | Claims that others have already complied or benefited. | "migliaia di utenti", "altri clienti hanno già", "come molti di voi" |
| `reciprocity` | Offers something small in exchange for compliance. | "in cambio di", "se compili riceverai" |

## Contrast with DIDECO

DIDECO defines 20 intent categories, in English, on LLM-generated
spear-phishing emails. IFIT is deliberately narrower because:

1. **Italian corpus size.** Our annotated corpus is 950 documents
   (versus DIDECO's larger LLM-generated pool). 12 intents keeps
   per-intent support above 5 for 9 of 12 categories.
2. **Real SMS and forum.** IFIT is trained on real (non-LLM) SMS and
   forum text in addition to email. Real channels have less lexical
   variety than LLM-generated messages and need fewer categories.
3. **Use/mention distinction.** IFIT adds an explicit use/mention
   annotation layer. DIDECO does not. This distinction matters because
   forum posts about phishing are stylistically close to phishing
   emails and account for a large fraction of false positives in naive
   classifiers.

## Annotation statistics (950 documents)

Per-intent support over the annotated corpus:

| Intent | Support |
|---|---:|
| `click_request` | 465 |
| `urgency` | 411 |
| `authority` | 244 |
| `impersonation` | 204 |
| `fear` | 199 |
| `credential_request` | 179 |
| `payment_request` | 121 |
| `data_request` | 105 |
| `greed` | 42 |
| `call_request` | 24 |
| `social_proof` | 3 |
| `reciprocity` | 3 |
| **Mean intents/doc (positive only)** | **2.97** |

## Cross-channel distribution

| Channel | Total | With intents | NONE |
|---|---:|---:|---:|
| email | 421 | 277 | 144 |
| sms | 416 | 352 | 64 |
| forum | 113 | 45 | 68 |
| **Total** | **950** | **674** | **276** |
