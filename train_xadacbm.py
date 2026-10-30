"""
Train X-AdaCBM (cross-modal concept attention adapter) on HAM10000.

Data discipline
  * trains on the TRAIN split features only
  * validates on the VAL split every epoch; best checkpoint + early stopping use VAL only
  * never loads the test split (evaluate_models.py does that, once, at the end)

Prerequisites (see notebooks/kaggle_run.ipynb)
  1. scripts/extract_features.py           -> <results>/features/<clip>/{train,val,test}_feats.pt
  2. main.py --run-name adacbm_ham10000    -> baseline AdaCBM run (concept selection + checkpoint)
"""
import argparse
import csv
import json
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

import compat  # noqa: F401
from compat import Config
import features
import model_io
import project_paths as pp
from models.asso_opt.asso_xadacbm import XAdaCBMConcept

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, x, y, bs=1024):
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(x), bs):
        logits, _, _ = model(x[i:i + bs])
        loss_sum += torch.nn.functional.cross_entropy(logits, y[i:i + bs], reduction="sum").item()
        correct += (logits.argmax(-1) == y[i:i + bs]).sum().item()
    return loss_sum / len(x), 100.0 * correct / len(x)


def main():
    ap = argparse.ArgumentParser(description="Train X-AdaCBM (train/val only)")
    ap.add_argument("--baseline-run", default="adacbm_ham10000", help="run that holds the concepts + AdaCBM checkpoint")
    ap.add_argument("--baseline-ckpt", default=None, help="AdaCBM checkpoint (default: <baseline run>/checkpoints/last.ckpt)")
    ap.add_argument("--run-name", default="xadacbm_ham10000")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10, help="early stopping on val accuracy")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    pl.seed_everything(args.seed)
    root = pp.results_root(args.results_dir)
    base_run = pp.run_dir(root, args.baseline_run)
    concepts_dir = base_run / "concepts"
    base_ckpt = Path(args.baseline_ckpt) if args.baseline_ckpt else base_run / "checkpoints" / "last.ckpt"
    if not base_ckpt.exists():
        raise FileNotFoundError(f"AdaCBM checkpoint not found: {base_ckpt}. Train the baseline first (main.py).")

    # Architecture / feature settings are inherited from the AdaCBM checkpoint, not re-typed here.
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base_cfg = Config(dict(ck["hyper_parameters"]["cfg"]))
    cfg = Config({k: base_cfg[k] for k in ("clip_model", "num_cls", "num_concept", "num_layers",
                                           "use_img_norm", "use_txt_norm", "use_residual",
                                           "use_rand_init", "init_val") if k in base_cfg})
    cfg.update(dataset="HAM10000", concept_path=base_cfg["concept_path"],
               lr=args.lr, max_epochs=args.epochs)
    cfg = pp.apply_to_cfg(cfg, run_name=args.run_name, results_dir=args.results_dir)
    cfg.data_root = str(concepts_dir)  # concepts are READ from the baseline run

    # --- data: train + val only ------------------------------------------------------
    fdir = pp.feature_dir(root, cfg.clip_model)
    tr = features.load_cached_split("train", fdir, cfg.clip_model)
    va = features.load_cached_split("val", fdir, cfg.clip_model)
    def prep(d):
        f = d["feats"].float()
        if cfg.use_img_norm:
            f = f / f.norm(dim=-1, keepdim=True)
        return f.to(device), d["labels"].long().to(device)
    xtr, ytr = prep(tr)
    xva, yva = prep(va)
    print(f"train {len(xtr)} | val {len(xva)} | concepts {cfg.num_concept} | device {device}")
    assert len(xtr) == pp.EXPECTED_SPLIT_SIZES["train"] and len(xva) == pp.EXPECTED_SPLIT_SIZES["val"]

    # --- model -----------------------------------------------------------------------
    model = XAdaCBMConcept(cfg).to(device)
    model.load_from_adacbm_checkpoint(base_ckpt)
    model.to(device)

    # Safety check: with gate == 0 and attn_blend == 0, X-AdaCBM must equal AdaCBM.
    ada = model_io.load_adacbm(base_ckpt, concepts_dir, device)
    with torch.no_grad():
        model.eval()
        diff = (model(xva[:256])[0] - ada(xva[:256])[0]).abs().max().item()
    if diff > 1e-4:
        raise RuntimeError(f"X-AdaCBM at init differs from AdaCBM by {diff:.2e}; wrong concept dir/checkpoint?")
    init_loss, init_acc = evaluate(model, xva, yva)
    print(f"init (== AdaCBM) val loss {init_loss:.4f} acc {init_acc:.2f}% | max|logit diff| {diff:.2e}")

    optimizer = model.configure_optimizers()
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)

    ckpt_dir = Path(cfg.work_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    mdir = Path(cfg.metrics_dir); mdir.mkdir(parents=True, exist_ok=True)
    log_f = open(mdir / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "gate", "attn_blend"])

    def save(path, epoch, vloss, vacc):
        torch.save({"state_dict": model.state_dict(),
                    "hyper_parameters": {"cfg": {k: v for k, v in cfg.items()}},
                    "epoch": epoch, "val_loss": vloss, "val_acc": vacc,
                    "baseline_ckpt": str(base_ckpt),
                    "train_fingerprint": features.split_fingerprint(tr["img_names"], tr["labels"].tolist()),
                    "val_fingerprint": features.split_fingerprint(va["img_names"], va["labels"].tolist())}, path)

    best = (-1.0, float("inf")); best_epoch = 0; bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = tc = n = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits, _, _ = model(bx)
            loss = torch.nn.functional.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(bx); tc += (logits.argmax(-1) == by).sum().item(); n += len(bx)
        vloss, vacc = evaluate(model, xva, yva)
        g = torch.tanh(model.attention_block.cross_attn.gate).item()
        ab = torch.tanh(model.attn_blend).item()
        print(f"epoch {epoch:3d} | train loss {tl/n:.4f} acc {100*tc/n:6.2f}% | val loss {vloss:.4f} acc {vacc:6.2f}% | gate {g:+.4f} blend {ab:+.4f}")
        log.writerow([epoch, tl / n, 100 * tc / n, vloss, vacc, g, ab]); log_f.flush()
        save(ckpt_dir / "last.ckpt", epoch, vloss, vacc)
        if (vacc, -vloss) > (best[0], -best[1]):
            best = (vacc, vloss); best_epoch = epoch; bad = 0
            save(ckpt_dir / "best.ckpt", epoch, vloss, vacc)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (no val-acc gain for {args.patience} epochs)")
                break
    log_f.close()
    summary = {"run": args.run_name, "baseline_run": args.baseline_run, "best_epoch": best_epoch,
               "best_val_acc": best[0], "best_val_loss": best[1], "init_val_acc": init_acc,
               "optimizer": "AdamW", "lr": args.lr, "seed": args.seed, "test_split_used": False}
    json.dump(summary, open(mdir / "train_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"best checkpoint: {ckpt_dir / 'best.ckpt'}  (selected on VALIDATION accuracy)")


if __name__ == "__main__":
    main()
