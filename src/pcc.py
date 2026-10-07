"""Post-hoc Channel Centering (PCC).

Frozen encoder. For each channel, subtract the per-channel mean CLS
vector from every embedding. Reduces Domain MI by ~55% while preserving
the intent signal (paper Table 8 / Table 12).

CLI:
    python -m src.pcc --checkpoint models/cicl_main.pt \
                      --data data/df_cicl.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .model import CICLMMDv2
from .utils import get_cls_embeddings, domain_mi_cv


def apply_pcc(embs, channels):
    """Subtract per-channel mean from every embedding."""
    centered = embs.copy()
    for c in np.unique(channels):
        mask = channels == c
        centered[mask] -= embs[mask].mean(axis=0, keepdims=True)
    return centered


def evaluate_pcc(model, df, tokenizer, device):
    """Return dict with MI and k-NN proxy before/after centering."""
    embs = get_cls_embeddings(model, df, tokenizer, device)
    channels = df['channel_id'].values

    mi_raw, _ = domain_mi_cv(embs, channels)
    embs_c = apply_pcc(embs, channels)
    mi_centered, _ = domain_mi_cv(embs_c, channels)

    return {
        'mi_raw':      mi_raw,
        'mi_centered': mi_centered,
        'reduction':   (mi_raw - mi_centered) / mi_raw,
    }


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model-name',
                        default='Musixmatch/umberto-commoncrawl-cased-v1')
    parser.add_argument('--n-intents', type=int, default=12)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = CICLMMDv2(args.model_name, args.n_intents).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    df = pd.read_csv(args.data)
    df['multihot'] = df['multihot'].apply(
        lambda s: np.array(eval(s), dtype=np.float32)
        if isinstance(s, str) else np.asarray(s, dtype=np.float32))

    results = evaluate_pcc(model, df, tokenizer, device)
    print(f"Domain MI (raw):      {results['mi_raw']:.4f}")
    print(f"Domain MI (centered): {results['mi_centered']:.4f}")
    print(f"Reduction:            {results['reduction']*100:.1f}%")


if __name__ == '__main__':
    _main()
