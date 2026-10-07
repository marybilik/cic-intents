"""Hybrid splitting for CIC-Intents.

Three independent splits, all reproducible with seed=42:
  MAIN       -- 70/15/15 stratified by channel and intent presence
  LOO-FORUM  -- train email+sms, test ALL forum (n=113)
  CV-FORUM   -- 5-fold stratified on forum, each fold adds email+sms
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

FORUM_CH = 3
RANDOM_STATE = 42


def make_splits(df_cicl):
    """Return a dict with 'main', 'loo', 'cv' split metadata."""
    df_forum   = df_cicl[df_cicl['channel_id'] == FORUM_CH].reset_index(drop=True)
    df_noforum = df_cicl[df_cicl['channel_id'] != FORUM_CH].reset_index(drop=True)

    # MAIN
    df_tr_main, df_tmp_main = train_test_split(
        df_cicl, test_size=0.30,
        stratify=df_cicl['subset'],
        random_state=RANDOM_STATE)
    df_va_main, df_te_main = train_test_split(
        df_tmp_main, test_size=0.50,
        stratify=df_tmp_main['subset'],
        random_state=RANDOM_STATE)

    # LOO-FORUM
    df_tr_loo, df_va_loo = train_test_split(
        df_noforum, test_size=0.15,
        stratify=df_noforum['subset'],
        random_state=RANDOM_STATE)
    df_te_loo = df_forum.copy()

    # CV-FORUM
    strat_label = (df_forum['n_intents'] > 0).astype(int).values
    unique, counts = np.unique(strat_label, return_counts=True)
    n_splits = min(5, counts.min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    cv_folds = list(skf.split(df_forum, strat_label))

    return {
        'main': dict(train=df_tr_main, val=df_va_main, test=df_te_main),
        'loo':  dict(train=df_tr_loo,  val=df_va_loo,  test=df_te_loo),
        'cv':   dict(folds=cv_folds, forum=df_forum, noforum=df_noforum),
    }
