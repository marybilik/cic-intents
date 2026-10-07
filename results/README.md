# Results directory

Machine-readable metrics and derived outputs from the CIC-Intents study.
Every number in `paper/main.tex` is traced to one of these files.

## Files

| File | Description |
|---|---|
| `paper_metrics.json` | All headline numbers as a single JSON dict |
| `ablation_study.csv` | Five-configuration ablation on the main split |
| `baseline_encoders.csv` | UmBERTo vs XLM-R vs mDeBERTa comparison |
| `lambda_sweep.csv` | λ_MMD sweep (Pareto frontier) |
| `forum_cv_results.csv` | Forum 5-fold CV per-fold metrics |
| `causal_attribution.csv` | PN/PS per (example, k) |
| `errors_fixed.csv` | Error taxonomy per test example |
| `bootstrap_ci.json` | 95% CIs for mAP, Macro-F1, Top-5 |
| `paired_tests.json` | Paired bootstrap tests vs alternative encoders |

## `paper_metrics.json` schema

```json
{
  "main_split": {
    "n_test": 143,
    "mAP_macro": 0.723,
    "macro_f1": 0.464,
    "top1_recall": 0.278,
    "top2_recall": 0.502,
    "top3_recall": 0.677,
    "top5_recall": 0.924,
    "precision_at_0.5": 0.365,
    "recall_at_0.5": 0.667,
    "exact_match": 0.140,
    "per_channel": {
      "email": {"n": 63, "f1": 0.452, "map": 0.608},
      "sms":   {"n": 63, "f1": 0.453, "map": 0.644},
      "forum": {"n": 17, "f1": 0.151, "map": 0.512}
    }
  },
  "forum_regimes": {
    "holdout": {"n": 17, "f1": 0.151, "map": 0.512},
    "cv5": {"n": 113, "f1_mean": 0.255, "f1_std": 0.062,
            "map_mean": 0.528, "map_std": 0.051},
    "loo": {"n": 113, "f1": 0.220, "map": 0.308}
  },
  "channel_invariance": {
    "raw_domain_mi": 0.739,
    "centered_domain_mi": 0.335,
    "reduction_pct": 54.7,
    "knn_intent_mean_abs_delta": 0.009
  },
  "causal_attribution": {
    "k_5":  {"mean_pn": 0.183, "mean_ps": 0.096},
    "k_10": {"mean_pn": 0.286, "mean_ps": 0.204},
    "k_20": {"mean_pn": 0.356, "mean_ps": 0.255}
  }
}
Reproducing these files
All results files are written by notebooks/cic_intents_full_pipeline.ipynb
at the end of training:

python
df_abl.to_csv(OUT_DIR + 'ablation_study.csv', index=False)
df_baselines.to_csv(OUT_DIR + 'baseline_encoders.csv', index=False)
df_sweep.to_csv(OUT_DIR + 'lambda_sweep.csv', index=False)
pd.DataFrame(forum_cv_rows).to_csv(OUT_DIR + 'forum_cv_results.csv', index=False)
df_pnps.to_csv(OUT_DIR + 'causal_attribution.csv', index=False)
df_errors.to_csv(OUT_DIR + 'errors_fixed.csv', index=False)
with open(OUT_DIR + 'paper_metrics.json', 'w') as f:
    json.dump(paper_metrics, f, indent=2)
License
Under CC-BY-4.0 — see ../data/LICENSE.
