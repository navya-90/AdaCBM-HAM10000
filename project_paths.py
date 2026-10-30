"""
Central place for every filesystem location used by the project.

Torch-free on purpose, so ``scripts/check_setup.py`` runs before any heavy import.

Resolution order for each location: environment variable > default.
Nothing in the project should contain an absolute path or rely on the current
working directory; everything is derived from here.

Environment variables
---------------------
HAM_IMAGE_ROOT   folder that (recursively) contains the ISIC_*.jpg files
SPLIT_DIR        folder with class2images_{train,val,test}.p
RESULTS_DIR      root of all outputs (features, runs, checkpoints, metrics)
CLIP_WEIGHTS     optional local ViT-L-14.pt (otherwise downloaded by `clip`)
CLIP_CACHE_DIR   optional download cache for CLIP weights
HAM_METADATA     optional path to HAM10000_metadata.csv (leakage check only)
"""
import os
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_SIZES = {"train": 8010, "val": 1000, "test": 1005}
IMG_EXTS = (".jpg", ".jpeg")


def is_kaggle():
    return Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v else default


def results_root(override=None):
    if override:
        return Path(override).resolve()
    default = "/kaggle/working/results" if is_kaggle() else REPO_ROOT / "results"
    return Path(_env("RESULTS_DIR", default)).resolve()


def split_dir():
    return Path(_env("SPLIT_DIR", REPO_ROOT / "datasets" / "HAM10000" / "splits"))


def run_dir(results, run_name):
    return Path(results) / "runs" / run_name


def clip_tag(clip_model):
    return clip_model.replace("/", "-")


def feature_dir(results, clip_model):
    return Path(results) / "features" / clip_tag(clip_model)


# --------------------------------------------------------------------------- splits
def load_splits(sdir=None):
    """Return {split: {class_name: [image filename, ...]}} exactly as stored."""
    sdir = Path(sdir) if sdir else split_dir()
    out = {}
    for s in SPLITS:
        f = sdir / f"class2images_{s}.p"
        if not f.exists():
            raise FileNotFoundError(f"Split file missing: {f}")
        with open(f, "rb") as fh:
            out[s] = pickle.load(fh)
    return out


def ordered_split(cls2img, cls_names):
    """Flatten a split in the fixed order (class order of cls_names, file order
    inside each class). Returns (names, labels)."""
    if set(cls2img.keys()) != set(cls_names):
        raise ValueError(
            f"Split classes {sorted(cls2img)} differ from cls_names {sorted(cls_names)}")
    names, labels = [], []
    for ci, c in enumerate(cls_names):
        for n in cls2img[c]:
            names.append(n)
            labels.append(ci)
    return names, labels


def class_names_from_splits(splits):
    keys = [sorted(splits[s].keys()) for s in SPLITS]
    if not all(k == keys[0] for k in keys):
        raise ValueError("Class names differ between split files")
    return keys[0]  # alphabetical == order used by cls_names.npy


# --------------------------------------------------------------------------- images
def build_image_index(root):
    """filename -> full path, searching `root` recursively. HAM10000 on Kaggle is
    usually split over two folders (part_1/part_2); this does not care."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    index = {}
    for dp, _, files in os.walk(root, followlinks=True):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS) and fn not in index:
                index[fn] = os.path.join(dp, fn)
    return index


def find_image_root():
    """HAM_IMAGE_ROOT, else the local datasets folder, else the first
    /kaggle/input/* folder that contains ISIC_*.jpg. Returns Path or None."""
    env = _env("HAM_IMAGE_ROOT")
    if env:
        return Path(env)
    local = REPO_ROOT / "datasets" / "HAM10000" / "images"
    if local.exists() and any(local.iterdir()):
        return local
    ki = Path("/kaggle/input")
    if ki.exists():
        for sub in sorted(ki.iterdir()):
            for dp, _, files in os.walk(sub, followlinks=True):
                if any(f.startswith("ISIC_") and f.lower().endswith(IMG_EXTS) for f in files):
                    return sub
    return None


def resolve_images(names, index, what="images"):
    missing = [n for n in names if n not in index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(names)} {what} not found under the image root "
            f"(e.g. {missing[:5]}). Check HAM_IMAGE_ROOT / the attached Kaggle dataset.")
    return [index[n] for n in names]


# --------------------------------------------------------------------------- config
def apply_to_cfg(cfg, run_name=None, results_dir=None, work_dir=None):
    """Overwrite every path in `cfg` so that nothing depends on the cwd.
    Layout:  <results>/features/<clip>/{train,val,test}_feats.pt
             <results>/runs/<run>/{concept_source,concepts,checkpoints,logs,metrics}
    """
    root = results_root(results_dir)
    run_name = run_name or cfg.get("run_name") or "adacbm_ham10000"
    run = run_dir(root, run_name)
    cfg.results_root = str(root)
    cfg.run_name = run_name
    cfg.run_dir = str(run)
    cfg.feature_dir = str(feature_dir(root, cfg.clip_model))
    cfg.concept_root = str(run / "concept_source") + os.sep
    cfg.raw_sen_path = cfg.concept_root + "concepts_raw.npy"
    cfg.concept2cls_path = cfg.concept_root + "concept2cls.npy"
    cfg.cls_name_path = cfg.concept_root + "cls_names.npy"
    cfg.data_root = str(run / "concepts")
    cfg.work_dir = str(work_dir) if work_dir else str(run / "checkpoints")
    cfg.log_dir = str(run / "logs")
    cfg.metrics_dir = str(run / "metrics")
    cfg.img_split_path = str(split_dir())
    root_img = find_image_root()
    cfg.img_path = str(root_img) if root_img else ""
    cp = Path(cfg.concept_path)
    if not cp.is_absolute():
        cp = REPO_ROOT / cp
    if not cp.exists():  # stale name in old configs
        cp = REPO_ROOT / "gpt_concepts" / f"{cfg.dataset}.json"
    cfg.concept_path = str(cp)
    for d in (cfg.concept_root, cfg.data_root, cfg.work_dir, cfg.log_dir,
              cfg.metrics_dir, cfg.feature_dir):
        os.makedirs(d, exist_ok=True)
    return cfg
