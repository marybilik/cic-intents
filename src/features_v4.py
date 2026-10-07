"""33 hand-crafted channel-invariant intent features (V4).

21 generic phishing signals + 12 Italian-specific markers. Used as an
alternative / complementary representation to the UmBERTo encoder.
Full ablation in paper Section 4.3.
"""

import re


def extract_intent_features_v4(text_original, text_lemmatized=""):
    """Extract 33 features from raw + lemmatized text."""
    if not isinstance(text_original, str):
        text_original = ""
    if not isinstance(text_lemmatized, str):
        text_lemmatized = ""
    t, t_lemma = text_original.lower(), text_lemmatized.lower()

    def find(p):
        return int(bool(re.search(p, t)) or bool(re.search(p, t_lemma)))

    return {
        # --- V3 core (21) ---
        'direct_address': find(
            r'\b(gentile cliente|caro cliente|gentile utente|gentile signor|'
            r'gentile signora)\b'),
        'cta_click': find(
            r'\b(clicca qui|clicchi qui|clicca sul|segui il link|'
            r'apri il link|clicca il link)\b'),
        'cta_access': find(
            r'\b(accedi al|acceda al|accedi subito|acceda subito|'
            r'verifica il tuo|verifichi il suo|accedi qui)\b'),
        'urgency_deadline': find(
            r'\b(entro 24|entro 48|entro le prossime|scade oggi|'
            r'ultimo avviso|immediatamente|urgente)\b'),
        'account_state': find(
            r'\b(conto (è |e )?(stato |stat[oa] )?blocc|'
            r'conto (è |e )?(stato |stat[oa] )?sospes|'
            r'carta (è |e )?(stata |stat[oa] )?blocc|'
            r'carta (è |e )?(stata |stat[oa] )?sospes|'
            r'utenza (è |e )?(stata |stat[oa] )?blocc|'
            r'utenza (è |e )?(stata |stat[oa] )?sospes)\b'),
        'sms_object_state': find(
            r'\b(pacco|conto|carta|utenza|account|spedizione|bonifico|rimborso)\b'
            r'.{0,40}\b(trattenere|trattenuto|bloccare|bloccato|sospendere|'
            r'sospeso|fallire|fallito|in sospeso)\b'),
        'sms_imperative': find(
            r'\b(verificare|contattare|chiamare|richiamare|cliccare|accedere|'
            r'confermare|fornire|aggiornare|completare|risolvere|evitare|'
            r'controllare)\b'),
        'sms_action_required': find(
            r'\b(necessario|obbligatorio|richiesto|cruciale|importante|urgente)\b'),
        'sms_amount': find(r'€\s?\d+|\d+\s?euro|\d+[,.]\d+\s?euro'),
        'sms_bank_action': find(
            r'\b(banca|poste|unicredit|intesa|nexi|paypal|bnl|inps|'
            r'agenzia delle entrate|amazon|corriere|bartolini|dhl|gls)\b'),
        'sms_url_short': int('[url]' in t and len(t) < 300),
        'sms_phone': int(bool(re.search(
            r'\[phone\]|3\d{8,10}|\+\d{10,13}', t))),
        'first_person_received': find(
            r'\b(ho ricevuto|mi è arrivato|mi è arrivata|mi sono arrivat|'
            r'ho visto|mi hanno mandato)\b'),
        'warning_marker': find(
            r'\b(attenzione a|segnalo|segnalazione|allarme|state attenti|'
            r'fate attenzione|occhio a|diffidate)\b'),
        'report_marker': find(
            r'\b(campagna di|cert-agid|csirt|rilevato|individuato|'
            r'analizzato|sintesi|bollettino)\b'),
        'advice_marker': find(
            r'\b(non cliccate|non cliccare|mai cliccare|evitate di|consiglio|'
            r'suggerisco|bisogna stare|dovete stare|non aprite|mai aprire)\b'),
        'first_person_opinion': len(re.findall(
            r'\b(penso|credo|secondo me|a mio parere|ritengo|immagino)\b', t)),
        'discussion_marker': find(
            r'\b(come faccio|cosa devo|qualcuno sa|chi mi aiuta|'
            r'avete notizie|capitato anche a voi)\b'),
        'exclamation_count': text_original.count('!'),
        'question_count': text_original.count('?'),
        'has_url_placeholder': int('[url]' in t),

        # --- V4 additions (12) ---
        'bank_entity_v4': int(bool(re.search(
            r'\b(banco bpm|bpm|postepay|poste italiane|unicredit|intesa|'
            r'sanpaolo|nexi|bnl|mps|bper|credem|mediolanum|fineco|ing|'
            r'revolut|n26|hype|satispay|banca sella|crédit agricole)\b', t))),
        'suspicious_transaction': int(bool(re.search(
            r'\b(autorizzazione di spesa|addebito|addebitato|transazione|'
            r'operazione sospetta|movimento sospetto|'
            r'pagamento non autorizzato)\b', t))),
        'new_device_access': int(bool(re.search(
            r'\b(nuovo dispositivo|nuovo accesso|accesso anomalo|'
            r'dispositivo associato|dispositivo sconosciuto|sicurezza web|'
            r'verifica della sicurezza)\b', t))),
        'blocking_action': int(bool(re.search(
            r'\b(sblocco|sbloccare|sblocca|limitato|limitata|bloccato|'
            r'bloccata|sospensione|sospeso|sospesa|disattivato|disattivata)\b', t))),
        'follow_link': int(bool(re.search(
            r'\b(seguire il link|segui il link|seguire il portale|'
            r'raggiungi il link|raggiungi il portale|link correlato)\b', t))),
        'new_number_scam': int(bool(re.search(
            r'\b(nuovo numero|mio nuovo numero|questo nuovo numero|'
            r'puoi whatsapp|scrivimi su whatsapp|contattami su whatsapp|'
            r'salvarlo|salva il numero|ho cambiato numero)\b', t))),
        'gentile_generic': int(bool(re.search(r'\bgentile[,\s]', t))),
        'url_plus_action': int('[url]' in t and bool(re.search(
            r'\b(seguire|segui|clicca|accedi|acceda|verifica|conferma|'
            r'sblocca|sbloccare|attivare)\b', t))),
        'no_personal_name': int(not bool(re.search(
            r'\b(ciao|caro|carissimo|dott\.|dottor|ing\.|avv\.|sig\.|'
            r'signor|signora|prof\.)\b', t))),
        'formal_you_sms': len(re.findall(
            r'\b(sua|suo|sue|suoi|lei|la sua|il suo)\b', t)),
        'social_scam': int(bool(re.search(
            r'\b(facebook|instagram|whatsapp|telegram|tiktok|linkedin)\b.*'
            r'\b(dati|rubato|rubati|violazione|bloccato|verifica)\b', t))),
        'alarm_marker': int(bool(re.search(
            r'\b(attenzione!|attenzione:|allarme|emergenza|avviso importante)\b', t))),
    }


def apply_v4(df, text_col='text_clean', lemma_col='text_lemma'):
    """Apply extract_intent_features_v4 row-wise; return augmented DataFrame."""
    import pandas as pd
    feats = df.apply(
        lambda r: extract_intent_features_v4(
            r.get(text_col, ''), r.get(lemma_col, '')),
        axis=1
    ).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
