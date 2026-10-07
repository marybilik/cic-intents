"""CIC-Intents model and dataset.

CICLDataset   -- tokenizes (text, multihot, channel) triples.
CICLMMDv2     -- UmBERTo + intent head + projection head + MMD regularizer.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

from .losses import MultiLabelSupConLoss, channel_mmd_loss


class CICLDataset(Dataset):
    def __init__(self, texts, multihot, channels, tokenizer, max_len=160):
        self.enc = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_len, return_tensors='pt')
        self.y = torch.tensor(np.stack(multihot), dtype=torch.float32)
        self.c = torch.tensor(channels, dtype=torch.long)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item['intent'] = self.y[i]
        item['channel'] = self.c[i]
        return item

    def __len__(self):
        return len(self.c)


class CICLMMDv2(nn.Module):
    """UmBERTo encoder + intent head (12 sigmoid) + projection head + MMD.

    Args:
        model_name: HuggingFace model id (default: UmBERTo-CommonCrawl-Cased-v1).
        n_intents: number of IFIT intents (12).
        proj_dim: dimension of the contrastive projection head (128).
        dropout: dropout probability (0.3).
    """

    def __init__(self, model_name, n_intents, proj_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size

        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, n_intents)
        )
        self.projection = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, proj_dim)
        )

        # NOTE: alpha = 0.5 matches the paper (Eq. 3, Section 4.4).
        self.supcon = MultiLabelSupConLoss(temperature=0.07, channel_weight=0.5)

    def forward(self, input_ids, attention_mask,
                intent_labels=None, channel_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.intent_head(cls)
        proj = self.projection(cls)

        lc, lmmd = None, None
        if intent_labels is not None and channel_labels is not None:
            lc = self.supcon(proj, intent_labels, channel_labels)
            lmmd = channel_mmd_loss(cls, channel_labels)

        return logits, proj, lc, lmmd
