"""Load trained AdaCBM / X-AdaCBM checkpoints reproducibly (training script, evaluation, dashboard)."""
from pathlib import Path

import torch

import compat  # noqa: F401  (must precede unpickling of checkpoints containing a Config)
from compat import Config


def _hparams_cfg(ckpt):
    cfg = ckpt["hyper_parameters"]["cfg"]
    return Config(dict(cfg))


def load_adacbm(ckpt_path, concepts_dir, device):
    """AdaCBM Lightning checkpoint; concept files are read from `concepts_dir`
    (the data_root of the run that produced the checkpoint)."""
    from models.asso_opt.asso_our import OurConcept
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _hparams_cfg(ck)
    cfg.data_root = str(concepts_dir)
    m = OurConcept.load_from_checkpoint(str(ckpt_path), map_location=device,
                                        weights_only=False, cfg=cfg)
    return m.to(device).eval()


def load_xadacbm(ckpt_path, concepts_dir, device):
    """X-AdaCBM checkpoint written by train_xadacbm.py; strict state-dict load."""
    from models.asso_opt.asso_xadacbm import XAdaCBMConcept
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(dict(ck["hyper_parameters"]["cfg"]))
    cfg.data_root = str(concepts_dir)
    m = XAdaCBMConcept(cfg)
    m.load_state_dict(ck["state_dict"], strict=True)
    return m.to(device).eval()


def find_ckpt(directory, pattern):
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No checkpoint matching {pattern} in {directory}")
    return files[-1]
