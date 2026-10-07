"""Training loop for CIC-Intents."""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from .model import CICLDataset, CICLMMDv2

RANDOM_STATE = 42


def compute_pos_weight_v2(df_train, n_intents, clip=5.0):
    """pos_weight = neg/pos, clipped to [1, clip] to avoid F1 collapse."""
    counts = np.stack(df_train['multihot'].values)
    pos = counts.sum(axis=0)
    neg = len(counts) - pos
    pw = neg / np.maximum(pos, 1)
    pw = np.clip(pw, 1.0, clip)
    return torch.tensor(pw, dtype=torch.float32)


def train_cicl(df_train, df_val, tokenizer, model_name, n_intents,
               device, epochs=10, lr=5e-5,
               lambda_supcon=0.3, lambda_mmd=0.3,
               batch_size=32, max_len=160, seed=RANDOM_STATE, verbose=True):
    """Train CIC-Intents; return (model, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tr_ds = CICLDataset(
        df_train['text_clean'].values,
        df_train['multihot'].values,
        df_train['channel_id'].values,
        tokenizer, max_len=max_len)
    va_ds = CICLDataset(
        df_val['text_clean'].values,
        df_val['multihot'].values,
        df_val['channel_id'].values,
        tokenizer, max_len=max_len)

    weights = 1.0 + 2.0 * df_train['n_intents'].values.astype(float)
    sampler = WeightedRandomSampler(weights, num_samples=len(df_train),
                                    replacement=True)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler,
                       num_workers=2, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=64, num_workers=2, pin_memory=True)

    model = CICLMMDv2(model_name, n_intents).to(device)
    pos_weight = compute_pos_weight_v2(df_train, n_intents, clip=5.0).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(tr_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.1)

    history = {'bce': [], 'supcon': [], 'mmd': [], 'val_f1': []}
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        tb, ts, tm, nb = 0.0, 0.0, 0.0, 0
        for batch in tr_dl:
            opt.zero_grad()
            logits, _, lc, lmmd = model(
                batch['input_ids'].to(device, non_blocking=True),
                batch['attention_mask'].to(device, non_blocking=True),
                batch['intent'].to(device, non_blocking=True),
                batch['channel'].to(device, non_blocking=True))
            lb = bce(logits, batch['intent'].to(device, non_blocking=True))
            loss = lb + lambda_supcon * lc + lambda_mmd * lmmd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            tb += lb.item(); ts += lc.item(); tm += lmmd.item(); nb += 1

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in va_dl:
                logits, _, _, _ = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device))
                preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
                ys.append(batch['intent'].numpy())
        yv = np.vstack(ys)
        pv = np.vstack(preds)
        val_f1 = f1_score(yv, pv, average='macro', zero_division=0)

        history['bce'].append(tb / nb)
        history['supcon'].append(ts / nb)
        history['mmd'].append(tm / nb)
        history['val_f1'].append(val_f1)

        if verbose:
            elapsed = (time.time() - t0) / 60
            print(f"  ep {ep+1}/{epochs} | BCE={tb/nb:.4f} | "
                  f"SupCon={ts/nb:.4f} | MMD={tm/nb:.4f} | "
                  f"val_f1={val_f1:.4f} | {elapsed:.2f} min")

    return model, history
