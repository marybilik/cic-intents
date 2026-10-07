# The Italian Fraud Intent Taxonomy (IFIT)

A 12-intent multi-label schema for Italian fraud, adapted from the
DIDECO taxonomy (Popovic et al., LREC 2026) and grounded in Speech Act
Theory and persuasion psychology.

**Mean number of intents per fraud document: 2.97** (n=950 annotated
documents across email, SMS, and forum).

## Structure

IFIT splits intents into two families:

- **Explicit (5)** — what the attacker *asks for*: a direct speech act
  whose illocutionary force is a request.
- **Implicit (7)** — how the attacker *manipulates*: strategies that
  shape the recipient's perception without an overt request.

The split matters empirically: in leave-forum-out evaluation, explicit
intents transfer across channels substantially better than implicit ones
(see §5.9 of the paper). We attribute this to the use/mention
distinction: explicit requests are phrased almost identically across
channels, while implicit manipulations are refracted through the
register of the channel.

---

## Explicit intents (5)

### `credential_request`
**Definition:** Asks for login credentials, passwords, PINs, or access codes.
**Italian keywords:** `password`, `credenziali`, `PIN`, `codice di accesso`, `OTP`, `SPID`
**Prototypical example:** *"Confermi i tuoi dati di accesso al link seguente per evitare la sospensione dell'account?"*
**Cross-channel transfer (leave-forum-out AP):** 0.627

### `payment_request`
**Definition:** Asks for a payment, bank transfer, or credit card details.
**Italian keywords:** `pagamento`, `bonifico`, `carta di credito`, `IBAN`, `saldo`, `addebito`
**Prototypical example:** *"Completa il pagamento di 2,99€ per sbloccare il tuo pacco in giacenza."*
**Cross-channel transfer:** 0.510

### `data_request`
**Definition:** Asks for personal data (name, address, date of birth, tax code).
**Italian keywords:** `codice fiscale`, `data di nascita`, `indirizzo`, `documento`, `dati personali`
**Prototypical example:** *"Aggiorna i tuoi dati anagrafici entro 48 ore per evitare la chiusura del conto."*
**Cross-channel transfer:** 0.485

### `click_request`
**Definition:** Asks the user to click a link or download an attachment.
**Italian keywords:** `clicca`, `apri il link`, `scarica`, `allegato`, `verifica qui`
**Prototypical example:** *"Clicca qui per verificare la tua identità: [URL]"*
**Cross-channel transfer:** 0.458
**Main-split AP:** 0.813 (highest among all intents)

### `call_request`
**Definition:** Asks the user to call a phone number.
**Italian keywords:** `chiama`, `contatta il`, `numero verde`, `assistenza telefonica`
**Prototypical example:** *"Chiama il numero 800-XXX-XXX per sbloccare la carta."*
**Cross-channel transfer:** 0.185 (limited — call requests are rare in email and forum data)

---

## Implicit intents (7)

### `urgency`
**Definition:** Creates time pressure ("act now", "within 24 hours").
**Italian keywords:** `subito`, `entro 24 ore`, `immediatamente`, `urgente`, `scade`
**Prototypical example:** *"Azione richiesta entro 24 ore."*
**Main-split AP:** 0.758 — **the most over-predicted intent** (see error analysis §5.11)
**Cross-channel transfer:** 0.236 (poor)

### `authority`
**Definition:** Invokes a trusted institution or authority figure.
**Italian keywords:** `banca`, `Poste`, `Intesa`, `UniCredit`, `autorità`, `guardia di finanza`
**Prototypical example:** *"Comunicazione ufficiale da Intesa Sanpaolo."*
**Main-split AP:** 0.588
**Cross-channel transfer:** 0.457

### `fear`
**Definition:** Threatens negative consequences (account blocking, legal action, loss of funds).
**Italian keywords:** `bloccato`, `sospeso`, `denuncia`, `azione legale`, `perdita`, `multe`
**Prototypical example:** *"Il tuo conto verrà bloccato se non confermi i dati."*
**Cross-channel transfer:** 0.252 (poor)

### `greed`
**Definition:** Promises a reward, prize, refund, or unexpected gain.
**Italian keywords:** `vincita`, `premio`, `rimborso`, `bonus`, `gratis`, `omaggio`
**Prototypical example:** *"Hai vinto un iPhone 15! Clicca per riscattare il premio."*
**Cross-channel transfer:** 0.048 (worst transfer)

### `impersonation`
**Definition:** Pretends to be a known entity or person.
**Italian keywords:** `Gentile cliente`, `il tuo consulente`, `servizio clienti`, `assistenza ufficiale`
**Prototypical example:** *"Siamo il team di sicurezza di UniCredit."*
**Cross-channel transfer:** 0.117 (poor)

### `social_proof`
**Definition:** Claims that others have already complied or benefited.
**Italian keywords:** `migliaia di clienti`, `come molti altri`, `tutti hanno già`, `consigliato da`
**Prototypical example:** *"Oltre 10.000 clienti hanno già aggiornato i propri dati."*
**Main-split AP:** 0.007 (support=1 in test split — reported for completeness only)

### `reciprocity`
**Definition:** Offers something small in exchange for compliance.
**Italian keywords:** `in cambio`, `ti offriamo`, `per te un`, `piccolo gesto`
**Prototypical example:** *"In cambio della tua conferma riceverai un buono da 10€."*
**Main-split AP:** 0.017 (support=1 in test split)

---

## Notes on annotation

- Annotations produced with `openai/gpt-oss-120b` via the Groq API,
  zero-shot prompt, checkpointed every 25 documents.
- Full prompt in [`annotation_prompt.md`](annotation_prompt.md).
- Validated on a random 10% sample.
- Multilabel: each document may carry 0 (legitimate), 1, or more intents.

## Use/mention distinction

IFIT does **not** annotate whether a document is a *use* (commits fraud)
or a *mention* (talks about fraud). That distinction is captured
separately by the `use_mention` field in the corpus:

- `use` — the message itself is fraudulent (email fraud, SMS fraud)
- `mention` — the message discusses fraud (Telegram, forum)

This is the second axis of the corpus design and is essential for
avoiding the false-positive problem in forum-style channels.
