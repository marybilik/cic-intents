"""Losses for CIC-Intents.

- MultiLabelSupConLoss: supervised contrastive with intent overlap as
  positive criterion, cross-channel positive pairs receive higher weight.
- channel_mmd_loss: multi-bandwidth kernel MMD on the CLS embedding
  between every pair of channels present in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelSupConLoss(nn.Module):
    """Multi-label SupCon: positive pair iff intent sets overlap.

    Cross-channel positives get weight 1 + channel_weight.
    Default alpha = 0.5 matches the paper (Eq. 3, Section 4.4).
    """

    def __init__(self, temperature=0.07, channel_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.channel_weight = channel_weight

    def forward(self, features, intent_labels, channel_labels):
        device = features.device
        B = features.shape[0]
        features = F.normalize(features, dim=1)

        sim = torch.matmul(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=device)

        overlap = torch.matmul(intent_labels.float(),
                               intent_labels.float().T)
        positive = (overlap > 0) & ~self_mask

        ch_i = channel_labels.unsqueeze(1)
        ch_j = channel_labels.unsqueeze(0)
        cross_ch = (ch_i != ch_j).float()
        weights = 1.0 + self.channel_weight * cross_ch

        sim_masked = sim.masked_fill(self_mask, -1e9)
        log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)

        weighted = log_prob * positive.float() * weights
        n_pos = positive.sum(dim=1).clamp(min=1)
        return (-weighted.sum(dim=1) / n_pos).mean()


def channel_mmd_loss(cls_embs, channel_labels, sigmas=(1.0, 2.0, 4.0, 8.0)):
    """Kernel MMD on CLS embeddings between every pair of channels in batch.

    Returns a scalar tensor on cls_embs.device (0 if no valid pair).
    """
    device = cls_embs.device
    zero = torch.zeros((), device=device, dtype=cls_embs.dtype)

    unique_ch = channel_labels.unique()
    if len(unique_ch) < 2:
        return zero

    total = zero
    n_pairs = 0
    for i in range(len(unique_ch)):
        for j in range(i + 1, len(unique_ch)):
            x = cls_embs[channel_labels == unique_ch[i]]
            y = cls_embs[channel_labels == unique_ch[j]]
            if x.shape[0] < 2 or y.shape[0] < 2:
                continue
            XX = torch.cdist(x, x, p=2) ** 2
            YY = torch.cdist(y, y, p=2) ** 2
            XY = torch.cdist(x, y, p=2) ** 2
            mmd = zero
            for s in sigmas:
                k_xx = torch.exp(-XX / (2 * s ** 2))
                k_yy = torch.exp(-YY / (2 * s ** 2))
                k_xy = torch.exp(-XY / (2 * s ** 2))
                mmd = mmd + k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
            total = total + mmd / len(sigmas)
            n_pairs += 1

    if n_pairs == 0:
        return zero
    return total / n_pairs
