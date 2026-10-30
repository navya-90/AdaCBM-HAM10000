"""
End-to-end wiring test with a FAKE CLIP and synthetic class-coloured images (no GPU, no dataset).

    python tests/smoke_pipeline.py

Checks that extraction -> baseline training (main.py) -> test mode -> X-AdaCBM training ->
evaluation all run with the project's paths/caches, and that the test split is never used for
training or checkpoint selection. It says NOTHING about real accuracy.
"""
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ---- fake CLIP, installed before anything imports clip_loader's functions -----------------
import clip_loader


class FakeCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.P = torch.randn(3, 768, generator=g) * 3
        self.R = torch.randn(77, 768, generator=g)

    def encode_image(self, x):
        f = x.mean((2, 3)) @ self.P + 0.05 * torch.randn(len(x), 768)
        return 10 * f / f.norm(dim=-1, keepdim=True)          # realistic CLIP-like scale

    def encode_text(self, tok):
        f = (tok.float() / 49408) @ self.R
        return 10 * f / f.norm(dim=-1, keepdim=True)


def fake_preprocess(im):
    a = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)


def fake_load(name, device):
    return FakeCLIP().to(device), fake_preprocess


clip_loader.load_clip = fake_load
import utils  # noqa: E402
utils._load_clip = fake_load
import project_paths as pp  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="adacbm_smoke_"))
img_root, results = tmp / "images", tmp / "results"
os.environ.update(HAM_IMAGE_ROOT=str(img_root), RESULTS_DIR=str(results))
print("workdir:", tmp)

# ---- synthetic images named exactly like the real split files, colour = class ------------
splits = pp.load_splits()
cls_names = pp.class_names_from_splits(splits)
rng = np.random.RandomState(0)
palette = rng.randint(30, 225, size=(7, 3))
for s in pp.SPLITS:
    for c, names in splits[s].items():
        for i, n in enumerate(names):
            d = img_root / ("HAM10000_images_part_1" if i % 2 else "HAM10000_images_part_2")
            d.mkdir(parents=True, exist_ok=True)
            col = np.clip(palette[cls_names.index(c)] + rng.randint(-25, 25, 3), 0, 255).astype(np.uint8)
            Image.fromarray(np.tile(col, (8, 8, 1))).save(d / n, quality=95)
print("images written:", sum(1 for _ in img_root.rglob("*.jpg")))


def run(script, *argv):
    old = sys.argv
    sys.argv = [script, *argv]
    try:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        sys.argv = old


run("scripts/extract_features.py", "--batch-size", "256")
run("scripts/extract_features.py")                      # second call must hit the verified cache
run("main.py", "--cfg", "cfg/HAM10000/HAM10000_allshot_fac.py", "--run-name", "base", "--func", "our_main",
    "--cfg-options", "max_epochs=3", "seed=1", "num_concept=70", "lr=0.0005", "bs=256", "num_layers=2",
    "pearson_weight=0.9", "use_img_norm=False", "use_txt_norm=False", "use_residual=False",
    "concept_select_fn='our_selection'", "clip_model='ViT-L/14'")
base = results / "runs" / "base"
assert (base / "checkpoints" / "last.ckpt").exists() and (base / "concepts" / "select_idx.pth").exists()
run("main.py", "--cfg", "cfg/HAM10000/HAM10000_allshot_fac.py", "--run-name", "base", "--test", "--func", "our_main",
    "--cfg-options", f"ckpt_path='{base / 'checkpoints' / 'last.ckpt'}'", "num_concept=70", "bs=256", "num_layers=2",
    "pearson_weight=0.9", "use_img_norm=False", "use_txt_norm=False", "use_residual=False",
    "concept_select_fn='our_selection'", "clip_model='ViT-L/14'")
run("train_xadacbm.py", "--baseline-run", "base", "--run-name", "x", "--epochs", "3", "--patience", "2")
run("evaluate_models.py", "--baseline-run", "base", "--xadacbm-run", "x", "--split", "test")

summ = json.load(open(results / "runs" / "x" / "metrics" / "test" / "summary.json"))
assert summ["results"][2]["n"] == 1005
assert json.load(open(results / "runs" / "x" / "metrics" / "train_summary.json"))["test_split_used"] is False
for f in ("predictions.csv", "classification_report.txt", "confusion_matrix.csv", "metrics.json"):
    assert (results / "runs" / "x" / "metrics" / "test" / "xadacbm" / f).exists(), f
print("\nSMOKE TEST PASSED (synthetic data; accuracies are meaningless)")
print(json.dumps([{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in summ["results"]], indent=1))
