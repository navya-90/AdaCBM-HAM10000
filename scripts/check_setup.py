#!/usr/bin/env python
"""Pre-flight check. Torch-free, costs no GPU time. Run it before extracting features.

Verifies: split sizes (8,010/1,000/1,005) and disjointness; every split image exists under the
image root (any folder layout); CLIP weights hash (if CLIP_WEIGHTS is set); and, if
HAM10000_metadata.csv can be found, that split labels match the official diagnoses and
whether the splits share lesions (lesion_id leakage).
"""
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths as pp

DX = {"akiec": "actinic keratoses", "bcc": "basal cell carcinoma", "bkl": "benign keratosis-like lesions",
      "df": "dermatofibroma", "nv": "melanocytic nevi", "mel": "melanoma", "vasc": "vascular lesions"}
ok = True


def check(cond, msg):
    global ok
    print(("[ OK ] " if cond else "[FAIL] ") + msg)
    ok &= bool(cond)


def find_metadata():
    env = os.environ.get("HAM_METADATA")
    if env and Path(env).exists():
        return Path(env)
    root = pp.find_image_root()
    for base in [root, Path("/kaggle/input")] if root else [Path("/kaggle/input")]:
        if base and Path(base).exists():
            for dp, _, fs in os.walk(base):
                for f in fs:
                    if f.lower().startswith("ham10000_metadata") and f.lower().endswith(".csv"):
                        return Path(dp) / f
    return None


def main():
    print(f"Environment: {'Kaggle' if pp.is_kaggle() else 'local'} | results dir: {pp.results_root()}")
    splits = pp.load_splits()
    names = {s: [n for v in splits[s].values() for n in v] for s in pp.SPLITS}
    for s in pp.SPLITS:
        check(len(names[s]) == pp.EXPECTED_SPLIT_SIZES[s] and len(set(names[s])) == len(names[s]),
              f"{s}: {len(names[s])} unique images (expected {pp.EXPECTED_SPLIT_SIZES[s]})")
    check(not (set(names['train']) & set(names['val'])) and not (set(names['train']) & set(names['test']))
          and not (set(names['val']) & set(names['test'])), "splits are pairwise disjoint")
    try:
        cls = pp.class_names_from_splits(splits)
        check(len(cls) == 7, f"7 classes, identical across splits: {cls}")
    except ValueError as e:
        check(False, str(e))

    root = pp.find_image_root()
    check(root is not None, f"image root: {root}  (set HAM_IMAGE_ROOT if wrong)")
    if root:
        index = pp.build_image_index(root)
        allnames = sum(names.values(), [])
        missing = [n for n in allnames if n not in index]
        check(not missing, f"all {len(allnames)} split images found under the image root"
              + (f" -- MISSING {len(missing)}, e.g. {missing[:3]}" if missing else f" ({len(index)} jpgs indexed)"))

    w = os.environ.get("CLIP_WEIGHTS")
    if w:
        try:
            from clip_loader import expected_clip_sha, sha256_file
            check(sha256_file(w) == expected_clip_sha("ViT-L/14"), f"CLIP_WEIGHTS hash matches ViT-L/14: {w}")
        except Exception as e:  # clip package not installed yet, or bad file
            check(False, f"CLIP_WEIGHTS check failed: {e}")
    else:
        print("[info] CLIP_WEIGHTS not set: `clip` will download ViT-L/14 (~890 MB, needs internet).")

    md = find_metadata()
    if md:
        lesion, dx = {}, {}
        with open(md, newline="") as f:
            for r in csv.DictReader(f):
                lesion[r["image_id"] + ".jpg"] = r["lesion_id"]
                dx[r["image_id"] + ".jpg"] = DX[r["dx"]]
        wrong = [(s, n) for s in pp.SPLITS for c, ns in splits[s].items() for n in ns if dx.get(n) != c]
        check(not wrong, f"split labels agree with official metadata diagnoses ({len(wrong)} disagree)")
        by = {s: {lesion[n] for n in names[s] if n in lesion} for s in pp.SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = by[a] & by[b]
            n_img = sum(1 for n in names[b] if lesion.get(n) in shared)
            print(f"[info] lesions shared by {a} and {b}: {len(shared)} ({n_img} {b} images). "
                  f"{'-> possible leakage, see README' if shared else ''}")
    else:
        print("[info] HAM10000_metadata.csv not found: skipped label/lesion-leakage check "
              "(attach the metadata file or set HAM_METADATA).")
    print("\nSETUP OK" if ok else "\nSETUP FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
