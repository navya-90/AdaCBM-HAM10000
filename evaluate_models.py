"""
Evaluate Zero-shot CLIP, AdaCBM and X-AdaCBM on a held-out split (default: TEST, 1,005 images).

* Loads the TRAINED checkpoints (no manual gate/blend overrides).
* Uses cached features that are verified against the current split files.
* Saves everything needed to back a reported number: predictions, ground truth,
  classification report, confusion matrix, accuracy, macro / weighted F1.

Run this on the test split only after model selection on the validation split is finished.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import features
import model_io
import project_paths as pp
from clip_loader import load_clip

device = "cuda" if torch.cuda.is_available() else "cpu"


def metrics_for(name, y_true, probs, cls_names, img_names, out_dir):
    y_pred = probs.argmax(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(cls_names)))
    m = {"model": name, "n": int(len(y_true)),
         "accuracy": 100 * accuracy_score(y_true, y_pred),
         "balanced_accuracy": 100 * balanced_accuracy_score(y_true, y_pred),
         "macro_f1": 100 * f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
         "weighted_f1": 100 * f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)}
    rep = classification_report(y_true, y_pred, labels=labels, target_names=cls_names,
                                zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    json.dump({**m, "per_class": rep}, open(out_dir / "metrics.json", "w"), indent=2)
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=labels, target_names=cls_names, zero_division=0))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(cls_names), comments="")
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "pred"] + [f"p_{c}" for c in cls_names])
        for n, t, p, pr in zip(img_names, y_true, y_pred, probs):
            w.writerow([n, cls_names[t], cls_names[p]] + [f"{v:.6f}" for v in pr])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--baseline-run", default="adacbm_ham10000")
    ap.add_argument("--xadacbm-run", default="xadacbm_ham10000")
    ap.add_argument("--adacbm-ckpt", default=None, help="default: <baseline run>/checkpoints/last.ckpt")
    ap.add_argument("--xadacbm-ckpt", default=None, help="default: <xadacbm run>/checkpoints/best.ckpt")
    ap.add_argument("--clip-model", default="ViT-L/14")
    args = ap.parse_args()

    root = pp.results_root(args.results_dir)
    base_run, x_run = pp.run_dir(root, args.baseline_run), pp.run_dir(root, args.xadacbm_run)
    ada_ckpt = Path(args.adacbm_ckpt) if args.adacbm_ckpt else base_run / "checkpoints" / "last.ckpt"
    x_ckpt = Path(args.xadacbm_ckpt) if args.xadacbm_ckpt else x_run / "checkpoints" / "best.ckpt"
    for c in (ada_ckpt, x_ckpt):
        if not c.exists():
            raise FileNotFoundError(f"Checkpoint not found: {c}")

    d = features.load_cached_split(args.split, pp.feature_dir(root, args.clip_model), args.clip_model)
    feats, y, names, cls_names = d["feats"].float().to(device), d["labels"].numpy(), d["img_names"], d["cls_names"]
    print(f"Evaluating on '{args.split}': {len(y)} images, classes: {cls_names}")
    if args.split == "test":
        assert len(y) == pp.EXPECTED_SPLIT_SIZES["test"]

    concepts_dir = base_run / "concepts"
    ada = model_io.load_adacbm(ada_ckpt, concepts_dir, device)
    xada = model_io.load_xadacbm(x_ckpt, concepts_dir, device)
    f_in = feats / feats.norm(dim=-1, keepdim=True) if bool(ada.cfg.use_img_norm) else feats

    with torch.no_grad():
        ada_probs = torch.softmax(ada(f_in)[0], -1).cpu().numpy()
        x_probs = torch.softmax(xada(f_in)[0], -1).cpu().numpy()
        clip_model, _ = load_clip(args.clip_model, device)
        import clip
        tok = clip.tokenize([f"a dermatoscopy of {c}" for c in cls_names]).to(device)
        t = clip_model.encode_text(tok).float()
        t = t / t.norm(dim=-1, keepdim=True)
        zs_probs = torch.softmax(100.0 * (feats / feats.norm(dim=-1, keepdim=True)) @ t.t(), -1).cpu().numpy()

    out = x_run / "metrics" / args.split
    rows = [metrics_for("Zero-shot CLIP", y, zs_probs, cls_names, names, out / "zeroshot_clip"),
            metrics_for("AdaCBM", y, ada_probs, cls_names, names, out / "adacbm"),
            metrics_for("X-AdaCBM", y, x_probs, cls_names, names, out / "xadacbm")]
    json.dump({"split": args.split, "adacbm_ckpt": str(ada_ckpt), "xadacbm_ckpt": str(x_ckpt),
               "results": rows}, open(out / "summary.json", "w"), indent=2)
    print(f"\n{'Model':<18}{'Acc %':>8}{'BalAcc %':>10}{'MacroF1 %':>11}{'WtdF1 %':>9}")
    for r in rows:
        print(f"{r['model']:<18}{r['accuracy']:>8.2f}{r['balanced_accuracy']:>10.2f}{r['macro_f1']:>11.2f}{r['weighted_f1']:>9.2f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
