"""CIC-Intents: channel-invariant intent detection for Italian multi-channel fraud.

Modules:
    cleaning       -- text cleaning pipeline v7 (HTML, entities, homoglyphs)
    features_v4    -- 33 hand-crafted channel-invariant features
    losses         -- MultiLabelSupConLoss, channel_mmd_loss
    model          -- CICLDataset, CICLMMDv2
    splits         -- hybrid splitting (main / LOO-forum / 5-fold CV)
    train          -- training loop
    utils          -- prediction, embeddings, domain MI
    pcc            -- post-hoc channel centering
    attribution    -- gradient-based causal attribution (PN/PS)
    figures        -- paper figure generation
"""

__version__ = "1.0.0"
__author__ = "Mariia Bilikhodze"
__license__ = "MIT"
