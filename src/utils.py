"""Prediction, embedding extraction, and domain MI utilities."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader

from .model import CICLDataset


def predict_proba(model, df, tokenizer, device, max_len=160, batch_size=64):
    """Return (probs, labels, channels) numpy arrays."""
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


def domain_mi_cv(embs, channels, n_splits=5):
    """Domain MI proxy: 5-fold CV accuracy of a logistic-regression
    channel probe. Higher = more channel information in embeddings.
    Returns (mean_accuracy, per_fold_scores).
    """
    n_splits = min(n_splits, min(np.bincount(channels)))
    if n_splits < 2:
        return float('nan'), np.array([])
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
    scores = cross_val_score(clf, embs, channels, cv=n_splits,
                              scoring='accuracy')
    return float(scores.mean()), scores
