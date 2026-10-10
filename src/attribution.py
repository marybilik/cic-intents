"""Gradient-based causal attribution (PN / PS).

For each test example and each target intent t:
  PN(t) = f_t(x) - f_t(x \\ T_k(x))
  PS(t) = f_t(T_k(x)) - f_t(x \\ T_k(x))

where T_k(x) are the top-k tokens by gradient×input saliency.

Sweeps k in {5, 10, 20}. Produces df_pnps for Figure 9 and Figure 10.

CLI:
    python -m src.attribution --checkpoint models/cicl_main.pt \
                              --data data/df_cicl.csv \
                              --out results/pnps.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .model import CICLMMDv2
from .utils import parse_multihot


def _topk_tokens(model, input_ids, attention_mask, target_intent, k, device):
    """Return indices of top-k tokens by gradient×input saliency.

    Padding positions (attention_mask == 0) are masked out. Special
    tokens such as <s> (position 0) and </s> are not explicitly removed:
    the attention mask already excludes padding, and BOS/EOS do not carry
    intent information, so they rarely enter the top-k.
    """
    ids = input_ids.to(device).unsqueeze(0)
    mask = attention_mask.to(device).unsqueeze(0)

    emb_layer = model.encoder.get_input_embeddings()
    emb = emb_layer(ids).detach().requires_grad_(True)
    out = model.encoder(inputs_embeds=emb, attention_mask=mask)
    cls = out.last_hidden_state[:, 0, :]
    logits = model.intent_head(cls)
    score = logits[0, target_intent]
    score.backward()

    saliency = (emb.grad * emb).norm(dim=-1).squeeze(0)
    valid = mask.squeeze(0).bool()
    saliency = saliency.masked_fill(~valid, float('-inf'))

    topk = torch.topk(saliency, k).indices.cpu().tolist()
    return topk


def _prob_with_masked_tokens(model, input_ids, attention_mask, target_intent,
                              masked_positions, device, mask_token_id):
    ids = input_ids.clone()
    for pos in masked_positions:
        if pos < ids.numel():
            ids[pos] = mask_token_id
    with torch.no_grad():
        logits, _, _, _ = model(
            ids.to(device).unsqueeze(0),
            attention_mask.to(device).unsqueeze(0))
    return torch.sigmoid(logits[0, target_intent]).item()


def compute_pnps(model, df, tokenizer, device, ks=(5, 10, 20),
                  n_examples=42, seed=42):
    """Return DataFrame with per-example per-k PN and PS."""
    rng = np.random.RandomState(seed)
    model.eval()

    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id
    records = []

    sub = df[df['multihot'].apply(lambda v: np.asarray(v).sum() > 0)]
    if len(sub) > n_examples:
        sub = sub.sample(n=n_examples, random_state=seed).reset_index(drop=True)

    for _, row in sub.iterrows():
        enc = tokenizer(row['text_clean'], truncation=True,
                        max_length=160, padding='max_length',
                        return_tensors='pt')
        ids = enc['input_ids'][0]
        am  = enc['attention_mask'][0]

        intents = np.where(np.asarray(row['multihot']) > 0.5)[0]
        for t in intents:
            for k in ks:
                topk = _topk_tokens(model, ids, am, int(t), k, device)

                # f_t(x)
                f_x = _prob_with_masked_tokens(
                    model, ids, am, t, [], device, mask_token_id)
                # f_t(x \ T_k)
                f_no_topk = _prob_with_masked_tokens(
                    model, ids, am, t, topk, device, mask_token_id)
                # f_t(T_k): keep only top-k, mask the rest
                rest = [i for i in range(ids.numel())
                        if am[i].item() == 1 and i not in topk]
                f_only_topk = _prob_with_masked_tokens(
                    model, ids, am, t, rest, device, mask_token_id)

                pn = f_x - f_no_topk
                ps = f_only_topk - f_no_topk
                records.append({
                    'intent': int(t),
                    'k': k,
                    'PN': pn,
                    'PS': ps,
                })

    return pd.DataFrame(records)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='results/pnps.csv')
    parser.add_argument('--model-name',
                        default='Musixmatch/umberto-commoncrawl-cased-v1')
    parser.add_argument('--n-intents', type=int, default=12)
    parser.add_argument('--n-examples', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = CICLMMDv2(args.model_name, args.n_intents).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    df = pd.read_csv(args.data)
    df['multihot'] = df['multihot'].apply(
        lambda s: parse_multihot(s, args.n_intents))

    pnps = compute_pnps(model, df, tokenizer, device,
                        n_examples=args.n_examples)
    pnps.to_csv(args.out, index=False)
    print(f"Saved {len(pnps)} rows to {args.out}")
    print(pnps.groupby('k')[['PN', 'PS']].mean())


if __name__ == '__main__':
    _main()
