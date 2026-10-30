#!/usr/bin/env python
"""Extract CLIP image features for train / val / test, stored separately and fingerprinted.

    python scripts/extract_features.py                  # uses cached features if they are valid
    python scripts/extract_features.py --force          # re-extract everything
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

import features
import project_paths as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-model", default="ViT-L/14")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--image-root", default=None, help="overrides HAM_IMAGE_ROOT")
    args = ap.parse_args()

    root = pp.results_root(args.results_dir)
    fdir = pp.feature_dir(root, args.clip_model)
    img_root = args.image_root or pp.find_image_root()
    splits = pp.load_splits()
    cls_names = pp.class_names_from_splits(splits)
    print(f"image root: {img_root}\nfeature dir: {fdir}\nclasses: {cls_names}")
    for s in pp.SPLITS:
        feats, labels, names = features.get_split_features(
            s, splits[s], cls_names, fdir, args.clip_model, img_root=img_root,
            allow_extract=True, force=args.force, batch_size=args.batch_size)
        assert len(names) == pp.EXPECTED_SPLIT_SIZES[s], f"{s}: {len(names)} images"
        print(f"[{s}] {tuple(feats.shape)} ready -> {features.feature_file(fdir, s)}")
    print("done")


if __name__ == "__main__":
    main()
