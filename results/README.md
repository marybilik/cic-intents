# Results directory

Machine-readable metrics and derived outputs from the CIC-Intents study.
Every number in `paper/main.tex` is traced to one of these files.

## Files

| File | Description |
|---|---|
| `paper_metrics.json` | All headline numbers as a single JSON dict (canonical) |
| `ablation_study.csv` | Five-configuration ablation on the main split |
| `baseline_encoders.csv` | UmBERTo vs XLM-R vs mDeBERTa comparison |
| `lambda_sweep.csv` | λ_MMD sweep (Pareto frontier) |
| `forum_cv_results.csv` | Forum 5-fold CV per-fold metrics |
| `causal_attribution.csv` | PN/PS per (example, k) |
| `errors_fixed.csv` | Error taxonomy per test example |
| `bootstrap_ci.json` | 95% CIs for mAP, Macro-F1, Top-5 |
| `paired_tests.json` | Paired bootstrap tests vs alternative encoders |

## `paper_metrics.json` schema

This is the canonical schema. Any field not present here is not part of
the reported results.

```json
{
  "test_n": 143,
  "mAP_macro": 0.7209756294977184,
  "mAP_naive": 0.5920994411175138,
  "mean_AUC": 0.801672030822406,
  "top_k_recall": {
    "k1": 0.27491408934707906,
    "k2": 0.5017182130584192,
    "k3": 0.6804123711340206,
    "k5": 0.9312714776632303
  },
  "threshold_0.5": {
    "precision": 0.3683050201794899,
    "recall": 0.6654166314180219,
    "macro_f1": 0.46537487285939,
    "exact_match": 0.13286713286713286
  },
  "per_channel_f1": {
    "email": 0.45063140470108864,
    "sms": 0.45238379183063976,
    "forum": 0.13546176046176048
  },
  "per_intent_ap": {
    "credential_request": 0.6314043925491577,
    "payment_request": 0.79870845341972,
    "data_request": 0.8161014405762304,
    "click_request": 0.8319380181734652,
    "call_request": null,
    "urgency": 0.784025413624966,
    "authority": 0.6175497686272129,
    "fear": 0.7880558557969515,
    "greed": 0.49481713688610235,
    "impersonation": 0.7261801858256602,
    "social_proof": 0.008928571428571428,
    "reciprocity": 0.015384615384615385
  },
  "centering": {
    "mi_raw": 0.6761647424849038,
    "mi_centered": 0.11595325002576398
  },
  "forum_small_n_fix": {
    "holdout_main_n17": 0.13546176046176048,
    "loo_forum_n113": {
      "n_test": 113,
      "macro_f1": 0.2282135459234247,
      "mAP": 0.33755283539684655,
      "mAP_naive": 0.30774882653023117,
      "precision": 0.18038023383183566,
      "recall": 0.47972947191697196
    },
    "forum_cv_5fold": {
      "n_splits": 5,
      "n_forum_total": 113,
      "fixed_intents": 10,
      "macro_f1_mean": 0.2700088462441845,
      "macro_f1_std": 0.06385272047178642,
      "mAP_mean": 0.5044313395705478,
      "mAP_std": 0.08216655266247881
    }
  },
  "bootstrap_ci": {
    "mAP":       {"mean": 0.7293, "lo": 0.6449, "hi": 0.8060},
    "Macro-F1":  {"mean": 0.4622, "lo": 0.4187, "hi": 0.5042},
    "Top-5":     {"mean": 0.9317, "lo": 0.9010, "hi": 0.9565}
  },
  "paired_tests": [
    {"vs": "XLM-R",    "metric": "mAP",   "delta":  0.0260, "p": 0.176},
    {"vs": "XLM-R",    "metric": "Top-5", "delta":  0.0279, "p": 0.020},
    {"vs": "mDeBERTa", "metric": "mAP",   "delta":  0.0310, "p": 0.168},
    {"vs": "mDeBERTa", "metric": "Top-5", "delta":  0.0243, "p": 0.022}
  ],
  "encoder_comparison": [
    {"encoder": "UmBERTo",   "macro_f1": 0.4654, "mAP": 0.7210, "top5_recall": 0.9313},
    {"encoder": "XLM-R",     "macro_f1": 0.4601, "mAP": 0.6959, "top5_recall": 0.9038},
    {"encoder": "mDeBERTa",  "macro_f1": 0.5010, "mAP": 0.6871, "top5_recall": 0.9072}
  ],
  "ablation": [
    {"config": "BCE only",                  "macro_f1": 0.5161, "knn_intent": 0.8213, "domain_MI": 0.7883},
    {"config": "+SupCon",                   "macro_f1": 0.4913, "knn_intent": 0.8417, "domain_MI": 0.8282},
    {"config": "+SupCon+MMD",               "macro_f1": 0.4654, "knn_intent": 0.8431, "domain_MI": 0.6762},
    {"config": "+SupCon+MMD+Centering",     "macro_f1": 0.4654, "knn_intent": 0.8393, "domain_MI": 0.4422},
    {"config": "Centering only",            "macro_f1": 0.5161, "knn_intent": 0.8152, "domain_MI": 0.3800}
  ],
  "config": {
    "lr": 5e-05,
    "lambda_supcon": 0.3,
    "lambda_mmd": 0.3,
    "pos_weight_clip": 5.0,
    "temperature": 0.07,
    "channel_weight": 0.5,
    "epochs": 10,
    "batch_size": 32,
    "max_len": 160,
    "seed": 42
  }
}
```

> **Note on schema differences from earlier drafts.** An earlier version
> of this file used flat fields (`top1_recall`, `top2_recall`, …) and a
> nested `main_split` block. That schema does not match the actual JSON
> produced by notebook 02. The version above is authoritative. In
> particular:
> - `top_k_recall` is a **dict** with keys `k1`, `k2`, `k3`, `k5`.
> - `per_intent_ap` is a flat dict keyed by intent code; missing intents
>   (e.g. `call_request`) are `null`.
> - `centering.mi_raw` = 0.676, `centering.mi_centered` = 0.116
>   (NMI, train-means → test protocol).
> - `ablation` uses **NMI** values (0.788, 0.828, 0.676, 0.442, 0.380).

## Reproducing these files

All results files are written by
[`../notebooks/02-training-evaluation.ipynb`](../notebooks/02-training-evaluation.ipynb)
at the end of training:

```python
df_abl.to_csv(OUT_DIR + 'ablation_study.csv', index=False)
df_baselines.to_csv(OUT_DIR + 'baseline_encoders.csv', index=False)
df_sweep.to_csv(OUT_DIR + 'lambda_sweep.csv', index=False)
pd.DataFrame(forum_cv_rows).to_csv(OUT_DIR + 'forum_cv_results.csv', index=False)
df_pnps.to_csv(OUT_DIR + 'causal_attribution.csv', index=False)
df_errors.to_csv(OUT_DIR + 'errors_fixed.csv', index=False)
with open(OUT_DIR + 'paper_metrics.json', 'w') as f:
    json.dump(paper_metrics, f, indent=2)
```

## License

Under CC-BY-4.0 — see [`../data/LICENSE`](../data/LICENSE).
