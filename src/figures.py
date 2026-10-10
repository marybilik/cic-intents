"""Paper figure generation.

All figures from the paper can be regenerated from a trained checkpoint
and the annotated corpus. Run with:

    python -m src.figures --checkpoint models/cicl_main.pt \
                          --data data/df_cicl.csv \
                          --outdir results/figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer
import torch

from .model import CICLMMDv2
from .utils import predict_proba, get_cls_embeddings, parse_multihot
from .splits import make_splits

INTENT_CODES = [
    'credential_request', 'payment_request', 'data_request',
    'click_request', 'call_request',
    'urgency', 'authority', 'fear', 'greed',
    'impersonation', 'social_proof', 'reciprocity',
]
CH_NAMES = {0: 'email', 1: 'sms', 3: 'forum'}


def fig_training_curves(history, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ep = np.arange(1, len(history['bce']) + 1)

    axes[0, 0].plot(ep, history['bce'], 'o-', color='#3498DB', lw=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('BCE')
    axes[0, 0].set_title('A. Intent classification loss', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, history['supcon'], 's-', color='#9B59B6', lw=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('SupCon')
    axes[0, 1].set_title('B. Multi-label contrastive loss', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(ep, history['mmd'], '^-', color='#16A085', lw=2)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('MMD')
    axes[1, 0].set_title('C. Channel-invariance loss (λ=0.3)',
                         fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ep, history['val_f1'], 'd-', color='#27AE60', lw=2)
    axes[1, 1].axhline(0.5, color='gray', ls='--', alpha=0.5, label='F1=0.5')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Val macro-F1')
    axes[1, 1].set_title('D. Validation performance', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.suptitle('CIC-Intents — training dynamics',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def fig_tsne(embs, channels, labels, outpath):
    tsne = TSNE(n_components=2, perplexity=25, random_state=42,
                init='pca', learning_rate='auto')
    coords = tsne.fit_transform(embs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    CH_COLORS = {0: '#3498DB', 1: '#E74C3C', 3: '#9B59B6'}

    for ch_id, ch_name in CH_NAMES.items():
        mask = channels == ch_id
        if mask.sum() == 0:
            continue
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        c=CH_COLORS[ch_id], label=ch_name, alpha=0.7,
                        s=25, edgecolor='black', linewidth=0.3)
    axes[0].set_title('A. Embeddings by channel', fontweight='bold')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].legend(); axes[0].grid(alpha=0.3)

    dominant = np.array([np.argmax(r) if r.sum() > 0 else -1 for r in labels])
    palette_int = plt.cm.tab20(np.linspace(0, 1, 20))
    for idx, code in enumerate(INTENT_CODES):
        mask = dominant == idx
        if mask.sum() == 0:
            continue
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        c=[palette_int[idx]], label=code,
                        alpha=0.7, s=25, edgecolor='black', linewidth=0.3)
    axes[1].set_title('B. Embeddings by dominant intent', fontweight='bold')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def fig_per_intent_ci(y_true, y_proba, outpath, n_boot=1000):
    from sklearn.metrics import f1_score
    n = len(y_true)
    rng = np.random.RandomState(42)
    f1s = {c: [] for c in INTENT_CODES}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        for i, c in enumerate(INTENT_CODES):
            if y_true[idx, i].sum() == 0:
                continue
            pred = (y_proba[idx, i] > 0.5).astype(int)
            f1s[c].append(f1_score(y_true[idx, i], pred, zero_division=0))

    ci = {}
    for c in INTENT_CODES:
        if f1s[c]:
            ci[c] = (float(np.mean(f1s[c])),
                     float(np.percentile(f1s[c], 2.5)),
                     float(np.percentile(f1s[c], 97.5)))

    codes = sorted(ci.keys(), key=lambda x: ci[x][0], reverse=True)
    means = [ci[c][0] for c in codes]
    lows  = [ci[c][1] for c in codes]
    highs = [ci[c][2] for c in codes]

    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(codes))
    err_low = [m - l for m, l in zip(means, lows)]
    err_high = [h - m for m, h in zip(means, highs)]
    ax.barh(y_pos, means, xerr=[err_low, err_high],
            color='#E74C3C', edgecolor='black', alpha=0.8, capsize=4)
    for i, m in enumerate(means):
        ax.text(m + 0.02, i, f'{m:.2f}', va='center', fontsize=9)
    ax.set_yticks(y_pos); ax.set_yticklabels(codes, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1 (bootstrap 95% CI, n=1000)')
    ax.set_title('Per-intent F1 — CIC-Intents test', fontweight='bold')
    ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def fig_pn_vs_k(pnps_df, outpath):
    k_means  = pnps_df.groupby('k')['PN'].mean()
    ps_means = pnps_df.groupby('k')['PS'].mean()

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.plot(k_means.index, k_means.values, 'o-', color='#d62728',
            lw=2.4, markersize=12, label='Mean PN',
            markerfacecolor='white', markeredgewidth=2.4)
    ax.plot(ps_means.index, ps_means.values, 's-', color='#1f77b4',
            lw=2.4, markersize=12, label='Mean PS',
            markerfacecolor='white', markeredgewidth=2.4)
    ax.set_xlabel('top-$k$ attributed tokens')
    ax.set_ylabel('Mean PN / PS')
    ax.set_xticks([5, 10, 20])
    ax.set_title('PN and PS vs. number of attributed tokens',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def make_all_figures(checkpoint, data_path, outdir,
                     model_name='Musixmatch/umberto-commoncrawl-cased-v1',
                     n_intents=12,
                     pnps_path=None):
    """Generate paper figures from a trained checkpoint.

    Figures produced:
      fig2_tsne.png          -- t-SNE of CLS embeddings
      fig5_per_intent_ci.png -- per-intent F1 with bootstrap CIs
      fig10_pn_vs_k.png      -- PN/PS vs k (requires pnps_path)
    """
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = CICLMMDv2(model_name, n_intents).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    df = pd.read_csv(data_path)
    df['multihot'] = df['multihot'].apply(
        lambda s: parse_multihot(s, n_intents))
    df['channel_id'] = df['channel_id'].astype(int)
    df['n_intents']  = df['multihot'].apply(lambda v: int(v.sum()))

    splits = make_splits(df)
    test = splits['main']['test']

    probs, ys, chs = predict_proba(model, test, tokenizer, device)
    embs = get_cls_embeddings(model, test, tokenizer, device)

    fig_tsne(embs, chs, ys, os.path.join(outdir, 'fig2_tsne.png'))
    fig_per_intent_ci(ys, probs, os.path.join(outdir, 'fig5_per_intent_ci.png'))

    if pnps_path is not None and os.path.exists(pnps_path):
        pnps = pd.read_csv(pnps_path)
        fig_pn_vs_k(pnps, os.path.join(outdir, 'fig10_pn_vs_k.png'))

    print(f"Figures saved to {outdir}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/df_cicl.csv')
    parser.add_argument('--outdir', default='results/figures/')
    parser.add_argument('--pnps', default=None,
                        help='optional path to causal_attribution.csv')
    args = parser.parse_args()
    make_all_figures(args.checkpoint, args.data, args.outdir,
                     pnps_path=args.pnps)


if __name__ == '__main__':
    _main()
