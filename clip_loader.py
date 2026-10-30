"""Single entry point for loading CLIP, so weights come from the same place everywhere.

* CLIP_WEIGHTS set  -> load that local file (e.g. a Kaggle dataset). Its SHA-256 is
  checked against the hash embedded in the official OpenAI download URL.
* otherwise         -> `clip.load(name)` downloads into CLIP_CACHE_DIR (needs internet);
  `clip` verifies the same SHA-256 itself.
"""
import hashlib
import os


def expected_clip_sha(name):
    from clip import clip as clip_impl   # `_MODELS` lives in clip/clip.py, not in the package namespace
    if name not in clip_impl._MODELS:
        raise ValueError(f"Unknown CLIP model {name!r}; options: {list(clip_impl._MODELS)}")
    return clip_impl._MODELS[name].split("/")[-2]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_clip(name, device):
    """Returns (model, preprocess). Identity of the weights == expected_clip_sha(name)."""
    import clip
    want = expected_clip_sha(name)
    local = os.environ.get("CLIP_WEIGHTS")
    if local:
        if not os.path.isfile(local):
            raise FileNotFoundError(f"CLIP_WEIGHTS points to a missing file: {local}")
        got = sha256_file(local)
        if got != want:
            raise ValueError(
                f"CLIP weights {local} have SHA-256 {got}, expected {want} for {name}. "
                "Wrong model or corrupted/truncated file.")
        return clip.load(local, device=device)
    root = os.environ.get("CLIP_CACHE_DIR")
    return clip.load(name, device=device, download_root=root) if root else clip.load(name, device=device)
