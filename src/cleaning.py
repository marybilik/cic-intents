"""Text cleaning pipeline v7.

Strict HTML stripping, entity-tag normalization, URL/email/phone placeholder
substitution, Cyrillic homoglyph normalization, zero-width removal.
Applied to all four channels before training.

Order is fixed and matters:
  1. strict HTML tag removal (whitelist, not catch-all)
  2. entity tag normalization (<NAMED_ENTITY> -> [ENTITY], <URL> -> [URL])
  3. HTML entity decoding
  4. URL / email / phone placeholder substitution
  5. Cyrillic homoglyph handling (3 cases: full-text, mixed, single-char)
  6. zero-width + whitespace collapse + lowercase
"""

import re

# --- Regexes (compiled once at import) ---

KNOWN_HTML = (
    r'html|head|body|div|span|p|a|b|i|u|br|hr|img|table|tr|td|th|ul|ol|li|'
    r'script|style|link|meta|form|input|button|textarea|select|option|'
    r'h[1-6]|nav|header|footer|section|article|strong|em|blockquote|pre|'
    r'code|iframe|video|audio|source|canvas|svg|main|aside|figure|figcaption|'
    r'mark|small|sub|sup|del|ins|time|address|dl|dt|dd|caption|thead|tbody|'
    r'tfoot|col|colgroup|fieldset|legend|label|optgroup|datalist|output'
)

HTML_RE       = re.compile(rf'</?(?:{KNOWN_HTML})\b(?:\s+[^>]*=[^>]*)?>')
ENTITY_TAG_RE = re.compile(r'<([A-Za-z][A-Za-z0-9_]{1,49})>')
URL_RE        = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
EMAIL_RE      = re.compile(r'\S+@\S+\.\S+')
PHONE_RE      = re.compile(
    r'(?:\+39[\s\-\.]?)?(?:3\d{2}[\s\-\.]?\d{3}[\s\-\.]?\d{3,4}'
    r'|0\d{1,3}[\s\-\.]?\d{5,8})'
)
CYRILLIC_RE = re.compile(r'[\u0400-\u04FF\u0500-\u052F]')
WS_RE       = re.compile(r'\s+')
ZW_RE       = re.compile(r'[\u200b\u200c\u200d\ufeff\u2060]')

HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'і': 'i', 'ї': 'i', 'є': 'e', 'ґ': 'g', 'ѕ': 's', 'ј': 'j', 'ԁ': 'd',
    'ɡ': 'g', 'ⅰ': 'i', 'ⅼ': 'l', 'ⲟ': 'o', 'ⲁ': 'a',
}

HTML_ENTITIES = {
    '&amp;': ' & ', '&nbsp;': ' ', '&quot;': '"',
    '&lt;': ' < ', '&gt;': ' > ', '&#39;': "'", '&apos;': "'",
}

SPECIFIC_TAGS = {
    'url':           '[URL]',
    'email':         '[EMAIL]',
    'email_address': '[EMAIL]',
    'phone':         '[PHONE]',
    'phone_number':  '[PHONE]',
    'number':        '[NUMBER]',
    'cardinal':      '[NUMBER]',
    'ordinal':       '[NUMBER]',
    'quantity':      '[NUMBER]',
    'percent':       '[PERCENT]',
    'date':          '[DATE]',
    'time':          '[TIME]',
    'date_time':     '[DATE]',
    'money':         '[MONEY]',
}


def clean_text(text):
    """Return (cleaned_text, flags_dict).

    flags dict keys: had_html, had_url, had_email, had_phone, had_cyrillic,
    had_homoglyph, had_artifact, is_cyrillic_text, is_mixed_text, was_none.
    """
    if not isinstance(text, str):
        return "", {"was_none": True}

    flags = {k: False for k in [
        'had_html', 'had_url', 'had_email', 'had_phone', 'had_cyrillic',
        'had_homoglyph', 'had_artifact', 'is_cyrillic_text', 'is_mixed_text',
    ]}

    # 1. Strict HTML
    if HTML_RE.search(text):
        flags['had_html'] = True
        text = HTML_RE.sub(' ', text)

    # 2. Entity tags
    def _repl(m):
        tag = m.group(1).lower()
        return f" {SPECIFIC_TAGS.get(tag, '[ENTITY]')} "
    new = ENTITY_TAG_RE.sub(_repl, text)
    if new != text:
        flags['had_artifact'] = True
        text = new

    # 3. HTML entities
    for k, v in HTML_ENTITIES.items():
        if k in text:
            text = text.replace(k, v)

    # 4. URL / email / phone
    if URL_RE.search(text):
        flags['had_url'] = True
        text = URL_RE.sub(' [URL] ', text)
    if EMAIL_RE.search(text):
        flags['had_email'] = True
        text = EMAIL_RE.sub(' [EMAIL] ', text)
    if PHONE_RE.search(text):
        flags['had_phone'] = True
        text = PHONE_RE.sub(' [PHONE] ', text)

    # 5. Cyrillic
    cyr = CYRILLIC_RE.findall(text)
    if cyr:
        flags['had_cyrillic'] = True
        n_cyr = len(cyr)
        n_alpha = sum(c.isalpha() for c in text)
        pct = n_cyr / max(n_alpha, 1)
        if n_cyr >= 2 or pct >= 0.05:
            flags['is_cyrillic_text'] = True
        else:
            flags['had_homoglyph'] = True
            for k, v in HOMOGLYPH_MAP.items():
                text = text.replace(k, v)

    # 6. Zero-width + whitespace + lowercase
    text = ZW_RE.sub('', text)
    text = WS_RE.sub(' ', text).strip().lower()
    return text, flags
