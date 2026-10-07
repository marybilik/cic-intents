# Model checkpoints

Trained model checkpoints are **not stored in this Git repository** — the
combined size is ~3.5 GB, beyond what git comfortably handles. They are
hosted externally and linked below.

## Downloads

### Kaggle dataset (primary)

All checkpoints are available as a public Kaggle dataset:

> **`mariabilik/checks`** — https://www.kaggle.com/datasets/mariabilik/checks

To download programmatically:

```bash
# requires Kaggle API token in ~/.kaggle/kaggle.json
mkdir -p models && cd models
kaggle datasets download -d mariabilik/checks
unzip checks.zip && rm checks.zip
Or manually:

Open the Kaggle dataset page.

Click Download → unzip.

Place all .pt files in this directory (models/).

Files
File	Size	Framework	Description
cicl_main.pt	445 MB	CIC-Intents (UmBERTo)	Main model — trained on email + SMS + forum
cicl_loo.pt	445 MB	CIC-Intents (UmBERTo)	LOO-forum model — trained on email + SMS only
baseline_xlmr.pt	~1.1 GB	CIC-Intents (XLM-R-base)	Encoder comparison — same framework
baseline_mdeberta.pt	~1.1 GB	CIC-Intents (mDeBERTa-v3)	Encoder comparison — same framework
abl_bce_only.pt	445 MB	CIC-Intents (BCE-only)	Ablation: BCE loss, no SupCon, no MMD
abl_supcon.pt	445 MB	CIC-Intents (+SupCon)	Ablation: BCE + SupCon, no MMD
Total: ~3.5 GB.

Loading a checkpoint
The architecture expects CICLMMDv2(model_name, n_intents):

python
import torch
from transformers import AutoTokenizer

# from the notebook: CICLMMDv2 class definition

MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
N_INTENTS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = CICLMMDv2(MODEL_NAME, N_INTENTS).to(DEVICE)
model.load_state_dict(torch.load("models/cicl_main.pt", map_location=DEVICE))
model.eval()

enc = tokenizer(
    "Gentile cliente, confermi i dati di accesso entro 24 ore?",
    truncation=True, padding="max_length", max_length=160,
    return_tensors="pt",
).to(DEVICE)

with torch.no_grad():
    logits, projection, _, _ = model(**enc)

probs = torch.sigmoid(logits[0]).cpu().numpy()
Note for XLM-R / mDeBERTa checkpoints: these use their own
tokenizers (xlm-roberta-base, microsoft/mdeberta-v3-base) and their
forward pass may return token_type_ids which the model's forward()
does not accept. Drop it before calling:

python
enc.pop("token_type_ids", None)
Reproduction
See ../docs/reproducibility.md for
hyperparameters, seeds, and training time. All checkpoints were produced
on an NVIDIA Tesla T4 with seed 42.

License
Checkpoints are released under CC-BY-4.0 — see
../data/LICENSE. They are provided for research
use only. Deployment in a real fraud-detection pipeline requires
re-validation on the target distribution and appropriate human
oversight.
