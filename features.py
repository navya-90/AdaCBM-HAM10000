"""
Feature caching that cannot silently go stale.

Every cached tensor is saved next to a `<file>.meta.json`. On load, the metadata is
recomputed from the *current* splits / CLIP weights / concept list and compared; any
difference raises StaleCacheError instead of reusing old data.

Image features are stored once per split:
    <feature_dir>/{train,val,test}_feats.pt   (dict: feats, labels, img_names)
    <feature_dir>/{train,val,test}_feats.pt.meta.json
"""
import hashlib
import json
import os
from pathlib import Path

import torch

import project_paths as pp
from clip_loader import expected_clip_sha, load_clip

SCHEMA = 1


class StaleCacheError(RuntimeError):
    pass


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def split_fingerprint(names, labels):
    h = hashlib.sha256()
    for n, l in zip(names, labels):
        h.update(f"{n}:{int(l)}\n".encode())
    return h.hexdigest()


def _meta_path(path):
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def save_checked(obj, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    with open(_meta_path(path), "w") as f:
        json.dump(dict(meta, schema=SCHEMA), f, indent=2, sort_keys=True)


def load_checked(path, expected):
    """Load `path` only if its stored metadata equals `expected` (on the keys of
    `expected`). Returns None if the file does not exist; raises StaleCacheError on mismatch."""
    path = Path(path)
    if not path.exists():
        return None
    mp = _meta_path(path)
    if not mp.exists():
        raise StaleCacheError(f"{path} has no metadata file; cannot verify it. Delete it to regenerate.")
    with open(mp) as f:
        stored = json.load(f)
    expected = dict(expected, schema=SCHEMA)
    bad = {k: (stored.get(k), v) for k, v in expected.items() if stored.get(k) != v}
    if bad:
        raise StaleCacheError(
            f"Cache {path} is stale/incompatible. Mismatched fields (stored, expected): {bad}. "
            "Delete it (or run scripts/extract_features.py --force).")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------- image features
def feature_file(feature_dir, split):
    return Path(feature_dir) / f"{split}_feats.pt"


def expected_meta(split, names, labels, clip_model):
    return {
        "kind": "img_feats",
        "split": split,
        "n": len(names),
        "split_fingerprint": split_fingerprint(names, labels),
        "clip_model": clip_model,
        "clip_sha256": expected_clip_sha(clip_model),
    }


@torch.no_grad()
def extract_split_features(split, names, labels, index, clip_model, device,
                           batch_size=128, num_workers=2):
    from PIL import Image
    paths = pp.resolve_images(names, index, what=f"{split} images")
    model, preprocess = load_clip(clip_model, device)
    model.eval()

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                return preprocess(im.convert("RGB"))

    dl = torch.utils.data.DataLoader(DS(), batch_size=batch_size, num_workers=num_workers)
    out = []
    try:
        from tqdm import tqdm
        dl_iter = tqdm(dl, desc=f"CLIP features [{split}]")
    except ImportError:
        dl_iter = dl
    for x in dl_iter:
        out.append(model.encode_image(x.to(device)).float().cpu())
    feats = torch.cat(out)
    assert feats.shape[0] == len(names) and torch.isfinite(feats).all(), "bad features"
    meta = expected_meta(split, names, labels, clip_model)
    meta.update(feature_dim=int(feats.shape[1]), device=str(device),
                preprocess=repr(preprocess), torch=torch.__version__)
    return feats, meta


def get_split_features(split, cls2img, cls_names, feature_dir, clip_model,
                       img_root=None, allow_extract=True, force=False, device=None,
                       batch_size=128):
    """Return (feats, labels, names) for one split, from a verified cache or by extraction."""
    names, labels = pp.ordered_split(cls2img, cls_names)
    f = feature_file(feature_dir, split)
    want = expected_meta(split, names, labels, clip_model)
    if not force:
        cached = load_checked(f, want)  # raises if stale
        if cached is not None:
            assert cached["img_names"] == names, "cached image order differs from split"
            return cached["feats"], cached["labels"], names
    if not allow_extract:
        raise StaleCacheError(
            f"Features for split '{split}' not found at {f}. Run scripts/extract_features.py first.")
    if not img_root:
        raise FileNotFoundError("No image root available to extract features (set HAM_IMAGE_ROOT).")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    index = pp.build_image_index(img_root)
    feats, meta = extract_split_features(split, names, labels, index, clip_model, device,
                                         batch_size=batch_size)
    save_checked({"feats": feats, "labels": torch.tensor(labels), "img_names": names}, f, meta)
    return feats, torch.tensor(labels), names


def load_cached_split(split, feature_dir, clip_model, sdir=None):
    """Strict loader for training/eval scripts: never extracts, always verifies against
    the current split files. Returns dict(feats, labels, img_names, cls_names)."""
    splits = pp.load_splits(sdir)
    cls_names = pp.class_names_from_splits(splits)
    feats, labels, names = get_split_features(split, splits[split], cls_names, feature_dir,
                                              clip_model, allow_extract=False)
    return {"feats": feats, "labels": labels, "img_names": names, "cls_names": cls_names}


# ---------------------------------------------------------------------- generic helper
def concept_hash(concepts):
    return stable_hash([str(c) for c in concepts])
