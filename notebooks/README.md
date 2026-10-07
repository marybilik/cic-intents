## `cic_intents_full_pipeline.ipynb`

Self-contained Jupyter notebook that reproduces every result in the
paper. Runs on Kaggle (NVIDIA T4 GPU, ~4 hours end-to-end).

### Contents

The notebook is organized into numbered cells:

| Cell | Purpose |
|---|---|
| 1. Setup | Imports, seeds, GPU check |
| 2. Data loading | Loads E-PhishGen, Smishing-IMC25, MuLTa, forum |
| 3. Preprocessing | HTML strip, entity normalization, homoglyphs, spaCy lemmas |
| 4. Annotation | IFIT labeling via Groq API (or load pre-annotated) |
| 5. Splits | 70/15/15 stratified + LOO-forum + 5-fold forum |
| 6. Model | `CICLMMDv2` class: UmBERTo + intent head + projection + MMD |
| 7. Training utilities | `train_cicl`, `predict_proba`, `get_cls_embeddings`, `channel_center`, `domain_mi_cv` |
| 7b. Main model | Trains `cicl_main.pt` (email + SMS + forum) |
| 7c. LOO-forum model | Trains `cicl_loo.pt` (email + SMS only) |
| 8. Main evaluation | mAP, AUC, Top-k, per-channel, per-intent |
| 8b. LOO-forum evaluation | Three-regime forum comparison |
| 9. Encoder comparison | UmBERTo vs XLM-R vs mDeBERTa |
| 10. λ-sweep | Pareto frontier (Macro-F1 vs Domain MI) |
| 10b. Ablation study | Five component configurations |
| 10c. Bootstrap CIs | 95% CIs + paired tests |
| 11. Post-hoc centering | Raw vs centered CLS analysis |
| 12. Causal attribution | Gradient×input saliency, PN/PS sweep |
| 13. Figures | Generates all 11 figures in `paper/figures/` |
| 14. Export | Writes `results/*.json`, `results/*.csv` |

### Running

**On Kaggle** (recommended):

1. Create a new Notebook.
2. Upload `cic_intents_full_pipeline.ipynb`.
3. **Add data:**
   - Dataset `mariabilik/df-cicl` (preprocessed corpus, required)
   - Dataset `mariabilik/checks` (optional — pre-trained checkpoints)
4. Set accelerator: **GPU T4 ×1** or **P100**.
5. Enable internet (needed for Groq API and HuggingFace downloads).
6. Run all.

**Locally:**

```bash
pip install -r ../requirements.txt
jupyter lab cic_intents_full_pipeline.ipynb
Requires a GPU with ≥ 15 GB VRAM for training. Post-hoc analyses
(bootstrap, causal attribution, figures) run on CPU.

Environment variables
bash
export GROQ_API_KEY="gsk_..."      # for annotation (skip if using pre-annotated)
export KAGGLE_USERNAME="marybilik" # only if using kaggle CLI
export KAGGLE_KEY="..."            # same
Notes
All random operations use seed 42.

Checkpoints are written every epoch to /kaggle/working/.

Download the outputs via the Kaggle Output panel before the session ends
— Kaggle wipes /kaggle/working/ after ~12h of inactivity.

See ../docs/reproducibility.md for the full checklist.
