import json, sys
from pathlib import Path

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}

cells = [
md("""# AdaCBM / X-AdaCBM on Kaggle (GPU)
**Before running:** Settings → Accelerator = GPU, Internet = On; *Add Data* → attach a HAM10000 image dataset
(optionally also a dataset containing `HAM10000_metadata.csv`, and a dataset with `ViT-L-14.pt`).
Edit `REPO_URL` below. Run top to bottom; stop after the check cell if anything says FAIL."""),
code("""
import os, subprocess, sys, pathlib
REPO_URL = "https://github.com/<your-user>/<your-repo>.git"   # <-- EDIT
BRANCH   = "main"
RUN_BASE = "adacbm_ham10000"
RUN_X    = "xadacbm_ham10000"
SEED     = 777
os.chdir("/kaggle/working")
if not pathlib.Path("AdaCBM").exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, "AdaCBM"], check=True)
else:
    subprocess.run(["git", "-C", "AdaCBM", "pull"], check=True)
os.chdir("/kaggle/working/AdaCBM")
sys.path.insert(0, os.getcwd())
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
"""),
code("""!pip install -q -r requirements-kaggle.txt
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"""),
code("""
# Where is everything? (images and weights are auto-detected under /kaggle/input; override if needed)
os.environ["RESULTS_DIR"] = "/kaggle/working/results"
# os.environ["HAM_IMAGE_ROOT"] = "/kaggle/input/<your-ham10000-dataset>"
# os.environ["CLIP_WEIGHTS"]   = "/kaggle/input/<your-clip-dataset>/ViT-L-14.pt"   # optional; else downloaded (internet on)
!ls /kaggle/input
"""),
code("!python scripts/check_setup.py"),
md("## 1. Extract features (train / val / test stored separately, fingerprinted)"),
code("!python scripts/extract_features.py"),
md("## 2. Train the AdaCBM baseline (concept selection on train features, checkpoint monitor = validation accuracy)"),
code("!bash train.sh $RUN_BASE $SEED" if False else "!bash train.sh {RUN_BASE} {SEED}"),
md("## 3. Train X-AdaCBM (train + val only; the test split is not loaded)"),
code("!python train_xadacbm.py --baseline-run {RUN_BASE} --run-name {RUN_X} --epochs 50 --lr 0.001 --patience 10"),
md("## 4. Run the unit tests (gate-zero equivalence, intervention consistency, cache staleness)"),
code("!python -m pytest -q tests"),
md("## 5. Final evaluation on the held-out test split — do this once, after model selection on validation"),
code("!python evaluate_models.py --split test --baseline-run {RUN_BASE} --xadacbm-run {RUN_X}"),
code("""
import shutil
shutil.make_archive("/kaggle/working/results_export", "zip", "/kaggle/working/results/runs")   # small: no features
print("download /kaggle/working/results_export.zip from the Output tab")
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("kaggle_run.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
