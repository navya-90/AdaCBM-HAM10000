import json
import pytest
import torch

import features
import project_paths as pp
from compat import Config


@pytest.fixture(autouse=True)
def fake_clip_hash(monkeypatch):
    monkeypatch.setattr(features, "expected_clip_sha", lambda name: "sha-of-" + name)


def test_real_splits_are_intact():
    sp = pp.load_splits()
    names = {s: [n for v in sp[s].values() for n in v] for s in pp.SPLITS}
    assert {s: len(v) for s, v in names.items()} == pp.EXPECTED_SPLIT_SIZES
    assert not set(names["train"]) & set(names["val"])
    assert not set(names["train"]) & set(names["test"])
    assert not set(names["val"]) & set(names["test"])
    assert all(n.startswith("ISIC_") and n.endswith(".jpg") for s in names.values() for n in s)
    assert len(pp.class_names_from_splits(sp)) == 7


def test_image_index_across_two_folders(tmp_path):
    for sub, n in (("HAM10000_images_part_1", "ISIC_0000001.jpg"), ("part2/nested", "ISIC_0000002.jpg")):
        (tmp_path / sub).mkdir(parents=True)
        (tmp_path / sub / n).write_bytes(b"x")
    (tmp_path / "ISIC_0000001_segmentation.png").write_bytes(b"x")
    idx = pp.build_image_index(tmp_path)
    assert set(idx) == {"ISIC_0000001.jpg", "ISIC_0000002.jpg"}
    assert pp.resolve_images(["ISIC_0000002.jpg"], idx)[0].endswith("ISIC_0000002.jpg")
    with pytest.raises(FileNotFoundError):
        pp.resolve_images(["ISIC_9999999.jpg"], idx)


def test_ordered_split_rejects_class_mismatch():
    with pytest.raises(ValueError):
        pp.ordered_split({"a": ["1.jpg"], "b": ["2.jpg"]}, ["a", "c"])


def test_apply_to_cfg_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "res"))
    cfg = Config(dict(clip_model="ViT-L/14", dataset="HAM10000", concept_path="gpt_concepts/HAM10000.json"))
    cfg = pp.apply_to_cfg(cfg, run_name="r1")
    assert cfg.concept_path.startswith(str(pp.REPO_ROOT)) and (pp.REPO_ROOT / "gpt_concepts" / "HAM10000.json").exists()
    for k in ("data_root", "work_dir", "feature_dir", "metrics_dir", "log_dir", "concept_root"):
        assert cfg[k].startswith(str(tmp_path / "res")), k
    assert "ViT-L-14" in cfg.feature_dir


def _fake_split_cache(tmp_path, cls2img, cls_names, clip="ViT-L/14"):
    names, labels = pp.ordered_split(cls2img, cls_names)
    meta = features.expected_meta("train", names, labels, clip)
    obj = {"feats": torch.randn(len(names), 4), "labels": torch.tensor(labels), "img_names": names}
    features.save_checked(obj, features.feature_file(tmp_path, "train"), meta)
    return names


def test_cache_roundtrip_and_staleness(tmp_path):
    cls = ["a", "b"]
    c2i = {"a": ["1.jpg", "2.jpg"], "b": ["3.jpg"]}
    names = _fake_split_cache(tmp_path, c2i, cls)
    f, l, n = features.get_split_features("train", c2i, cls, tmp_path, "ViT-L/14", allow_extract=False)
    assert n == names and f.shape == (3, 4)
    # a different split (image moved to another class) must NOT reuse the cache
    with pytest.raises(features.StaleCacheError):
        features.get_split_features("train", {"a": ["1.jpg"], "b": ["2.jpg", "3.jpg"]}, cls, tmp_path,
                                    "ViT-L/14", allow_extract=False)
    # different image list
    with pytest.raises(features.StaleCacheError):
        features.get_split_features("train", {"a": ["1.jpg", "9.jpg"], "b": ["3.jpg"]}, cls, tmp_path,
                                    "ViT-L/14", allow_extract=False)
    # different CLIP weights
    with pytest.raises(features.StaleCacheError):
        features.get_split_features("train", c2i, cls, tmp_path, "ViT-B/32", allow_extract=False)


def test_missing_cache_and_missing_meta(tmp_path):
    c2i = {"a": ["1.jpg"]}
    with pytest.raises(features.StaleCacheError):          # strict mode never extracts
        features.get_split_features("val", c2i, ["a"], tmp_path, "ViT-L/14", allow_extract=False)
    _fake_split_cache(tmp_path, c2i, ["a"])
    features._meta_path(features.feature_file(tmp_path, "train")).unlink()
    with pytest.raises(features.StaleCacheError):          # un-verifiable file is rejected
        features.get_split_features("train", c2i, ["a"], tmp_path, "ViT-L/14", allow_extract=False)


def test_split_fingerprint_is_order_and_label_sensitive():
    f = features.split_fingerprint
    assert f(["a", "b"], [0, 1]) != f(["b", "a"], [1, 0])
    assert f(["a", "b"], [0, 1]) != f(["a", "b"], [1, 0])
