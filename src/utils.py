"""Utilities: prediction, CLS extraction, domain MI, multi-hot parsing."""

import ast
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from torch.utils.data import DataLoader

from .model import CICLDataset


# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
INTENT_CODES = [
    'credential_request', 'payment_request', 'data_request',
    'click_request', 'call_request',
    'urgency', 'authority', 'fear', 'greed',
    'impersonation', 'social_proof', 'reciprocity',
]
N_INTENTS = len(INTENT_CODES)
MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
CH_NAMES = {0: 'email', 1: 'sms', 3: 'forum'}
RANDOM_STATE = 42


# ----------------------------------------------------------------
# Paths (env-var driven; falls back to repo-local defaults)
# ----------------------------------------------------------------
def data_dir():
    return os.environ.get('CIC_DATA_DIR', './data/')


def ckpt_dir():
    return os.environ.get('CIC_CKPT_DIR', './models/')


def out_dir():
    return os.environ.get('CIC_OUT_DIR', './results/')


# ----------------------------------------------------------------
# Multi-hot parsing (safe: no eval)
# ----------------------------------------------------------------
def parse_multihot(s, n_intents=N_INTENTS):
    """Parse a multi-hot vector from a CSV cell.

    Accepts numpy arrays, python lists, strings like '[0. 1. 0.]',
    and strings like 'array([0., 1., 0.])'.
    """
    if isinstance(s, np.ndarray):
        return s.astype(np.float32).ravel()

    if isinstance(s, (list, tuple)):
        return np.asarray(s, dtype=np.float32)

    if isinstance(s, str):
        s = s.strip()
        if s.startswith('array(') and s.endswith(')'):
            s = s[len('array('):-1]
        # Replace whitespace-separated floats '0. 1. 0.' with commas
        if ',' not in s and ' ' in s:
            s = '[' + ','.join(s.strip('[]').split()) + ']'
        try:
            v = ast.literal_eval(s)
            arr = np.asarray(v, dtype=np.float32).ravel()
            if arr.shape[0] != n_intents:
                return np.zeros(n_intents, dtype=np.float32)
            return arr
        except (ValueError, SyntaxError):
            return np.zeros(n_intents, dtype=np.float32)

    return np.zeros(n_intents, dtype=np.float32)


def attach_multihot(df, col='multihot', n_intents=N_INTENTS):
    """Add/replace df['multihot'] with parsed numpy vectors, in place."""
    df[col] = df[col].apply(lambda s: parse_multihot(s, n_intents))
    return df


# ----------------------------------------------------------------
# Prediction and CLS extraction
# ----------------------------------------------------------------
def predict_proba(model, df, tokenizer, device, max_len=160, batch_size=64):
    """Return (probs, labels, channels) as numpy arrays."""
    ds = CICLDataset(
        df['text_clean'].values,
        df['multihot'].values,
        df['channel_id'].values,
        tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)
    model.eval()
    probs, ys, chs = [], [], []
    with torch.no_grad():
        for batch in dl:
            logits, _, _, _ = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(batch['intent'].numpy())
            chs.append(batch['channel'].numpy())
    return np.vstack(probs), np.vstack(ys), np.concatenate(chs)


def get_cls_embeddings(model, df, tokenizer, device, max_len=160,
                       batch_size=64):
    """Return (n, hidden) numpy array of CLS embeddings."""
    ds = CICLDataset(
        df['text_clean'].values,
        df['multihot'].values,
        df['channel_id'].values,
        tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in dl:
            out = model.encoder(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device))
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embs)


# ----------------------------------------------------------------
# Domain MI probe
# ----------------------------------------------------------------
def domain_mi_cv(embs, channels, n_splits=5, seed=RANDOM_STATE):
    """Domain MI proxy: NMI between channel labels and probe predictions.

    Returns (normalized_mutual_info_score, probe_accuracy).

    The paper reports **NMI** values everywhere (0.676 for raw CLS,
    0.116 for centered CLS, 0.442 for the full model under the
    test-means protocol, 0.380 for BCE-only under the same protocol).
    Using unnormalized `mutual_info_score` here would produce different
    numbers and break the trace from code to paper.

    n_splits is capped at the smallest non-empty class count, computed
    via np.unique (not np.bincount, which breaks when channel ids are
    not contiguous). LogisticRegression uses max_iter=3000 and
    class_weight='balanced'.
    """
    channels = np.asarray(channels).astype(int)
    embs = np.asarray(embs, dtype=np.float32)

    unique, counts = np.unique(channels, return_counts=True)
    n_splits = min(n_splits, int(counts.min()))
    if n_splits < 2:
        return float('nan'), float('nan')

    clf = LogisticRegression(
        max_iter=3000, class_weight='balanced', random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = cross_val_predict(clf, embs, channels, cv=skf)

    return (float(normalized_mutual_info_score(channels, preds)),
            float(accuracy_score(channels, preds)))
