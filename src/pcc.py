"""Post-hoc Channel Centering (PCC).

Frozen encoder. For each channel, subtract the per-channel mean CLS
vector from every embedding.

Two protocols are reported in the paper:
  * train-means -> test (headline, Table 12): per-channel means are
    estimated on the TRAINING split and applied to the test split.
    Reported: MI 0.676 -> 0.116 (-82.9%).
  * test-means (ablation, Table 8): per-channel means are estimated
    on the TEST batch itself. Reported: MI 0.676 -> 0.442 for the full
    model; MI 0.788 -> 0.380 for the BCE-only baseline.

CLI:
    python -m src.pcc --checkpoint models/cicl_main.pt \
                      --data data/df_cicl.csv \
                      --protocol train-means
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .model import CICLMMDv2
from .utils import get_cls_embeddings, domain_mi_cv, parse_multihot
from .splits import make_splits


def channel_means(embs, channels):
    """Return dict {channel_id: mean_vector} estimated on embs."""
    means = {}
    for c in np.unique(channels):
        mask = channels == c
        means[int(c)] = embs[mask].mean(axis=0, keepdims=False)
    return means


def apply_pcc(embs, channels, means=None):
    """Subtract per-channel mean from every embedding.

    Args:
        embs:      (n, d) array of CLS embeddings.
        channels:  (n,) array of channel ids.
        means:     optional dict {channel_id: mean_vector}. If None,
                   means are estimated on `embs` itself (test-means
                   protocol). To reproduce the headline number, pass
                   means computed on the TRAINING split.

    Returns:
        centered: (n, d) array, same shape as embs.
    """
    centered = np.asarray(embs, dtype=np.float32).copy()
    channels = np.asarray(channels).astype(int)

    if means is None:
        means = channel_means(centered, channels)

    for c, mu in means.items():
        mask = channels == c
        if mask.sum() == 0:
            continue
        centered[mask] = centered[mask] - mu
    return centered


def evaluate_pcc(model, df_test, tokenizer, device, df_train=None,
                 protocol='train-means'):
    """Evaluate PCC on df_test.

    Args:
        df_test:   DataFrame to evaluate on.
        df_train:  DataFrame from which to estimate per-channel means.
                   Required when protocol='train-means'.
        protocol:  'train-means' (paper headline) or 'test-means'
                   (ablation).
    """
    assert protocol in ('train-means', 'test-means'), \
        f"unknown protocol: {protocol}"

    embs_te = get_cls_embeddings(model, df_test, tokenizer, device)
    chs_te = df_test['channel_id'].values

    if protocol == 'train-means':
        if df_train is None:
            raise ValueError("protocol='train-means' requires df_train")
        embs_tr = get_cls_embeddings(model, df_train, tokenizer, device)
        chs_tr = df_train['channel_id'].values
        means = channel_means(embs_tr, chs_tr)
    else:
        means = None

    mi_raw, _ = domain_mi_cv(embs_te, chs_te)
    embs_c = apply_pcc(embs_te, chs_te, means=means)
    mi_centered, _ = domain_mi_cv(embs_c, chs_te)

    return {
        'protocol':    protocol,
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
    parser.add_argument('--protocol',
                        choices=['train-means', 'test-means'],
                        default='train-means',
                        help='train-means (paper headline) or '
                             'test-means (ablation)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = CICLMMDv2(args.model_name, args.n_intents).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    df = pd.read_csv(args.data)
    df['multihot'] = df['multihot'].apply(
        lambda s: parse_multihot(s, args.n_intents))
    df['channel_id'] = df['channel_id'].astype(int)
    df['n_intents']  = df['multihot'].apply(lambda v: int(v.sum()))

    if args.protocol == 'train-means':
        splits = make_splits(df)
        df_train = splits['main']['train']
        df_test = splits['main']['test']
    else:
        df_train, df_test = None, df

    results = evaluate_pcc(model, df_test, tokenizer, device,
                           df_train=df_train, protocol=args.protocol)
    print(f"Protocol:             {results['protocol']}")
    print(f"Domain MI (raw):      {results['mi_raw']:.4f}")
    print(f"Domain MI (centered): {results['mi_centered']:.4f}")
    print(f"Reduction:            {results['reduction']*100:.1f}%")


if __name__ == '__main__':
    _main()
