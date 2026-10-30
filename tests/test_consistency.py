"""Covers review issues 3, 5, 9, 10 on a tiny synthetic concept bank (no CLIP / data needed)."""
import numpy as np
import pytest
import torch

import compat  # noqa: F401
from compat import Config
import intervention
import model_io
from models.asso_opt.asso_our import OurConcept
from models.asso_opt.asso_xadacbm import XAdaCBMConcept

K, D, C = 70, 32, 7
CLS_OF = np.repeat(np.arange(C), K // C)


@pytest.fixture()
def cfg(tmp_path):
    torch.save(torch.randn(K, D), tmp_path / "concepts_feat_ViT-L-14.pth")
    np.save(tmp_path / "concepts_raw_selected.npy", np.array([f"concept {i}" for i in range(K)]))
    np.save(tmp_path / "concept2cls_selected.npy", CLS_OF)
    return Config(dict(data_root=str(tmp_path), clip_model="ViT-L/14", num_concept=K, num_cls=C,
                       num_layers=2, use_txt_norm=False, use_img_norm=False, use_residual=False,
                       use_rand_init=False, init_val=1.0, lr=1e-3, max_epochs=3))


def _randomised_adacbm(cfg):
    torch.manual_seed(0)
    m = OurConcept(Config(dict(cfg)))
    with torch.no_grad():
        m.asso_mat.add_(0.05 * torch.randn_like(m.asso_mat))
        m.asso_mat_bias.copy_(torch.randn(C))
        m.dot_product_bias.copy_(torch.randn(K))
    return m.eval()


def _x_from(ada, cfg, tmp_path):
    p = tmp_path / "ada.ckpt"
    torch.save({"state_dict": ada.state_dict()}, p)
    return XAdaCBMConcept(Config(dict(cfg))).load_from_adacbm_checkpoint(p).eval()


def test_zero_gate_reproduces_adacbm_exactly(cfg, tmp_path):                      # issue 3
    ada = _randomised_adacbm(cfg)
    x = _x_from(ada, cfg, tmp_path)
    feats = torch.randn(64, D)
    with torch.no_grad():
        la, ca = ada(feats)
        lx, cx, _ = x(feats)
    assert torch.allclose(la, lx, atol=1e-6) and torch.allclose(ca, cx, atol=1e-6)


def test_nonzero_gate_actually_changes_the_model(cfg, tmp_path):
    ada = _randomised_adacbm(cfg)
    x = _x_from(ada, cfg, tmp_path)
    with torch.no_grad():
        x.attention_block.cross_attn.gate.fill_(0.5)
        feats = torch.randn(8, D)
        assert not torch.allclose(ada(feats)[0], x(feats)[0], atol=1e-4)


def test_gate_receives_gradient_at_zero_init(cfg, tmp_path):
    x = _x_from(_randomised_adacbm(cfg), cfg, tmp_path).train()
    loss = torch.nn.functional.cross_entropy(x(torch.randn(16, D))[0], torch.randint(0, C, (16,)))
    loss.backward()
    assert x.attention_block.cross_attn.gate.grad.abs().item() > 0
    assert x.attn_blend.grad.abs().item() > 0


@pytest.mark.parametrize("which", ["adacbm", "xadacbm"])
def test_unmodified_intervention_reproduces_logits(cfg, tmp_path, which):        # issues 5 + 10
    ada = _randomised_adacbm(cfg)
    x = _x_from(ada, cfg, tmp_path)
    with torch.no_grad():                         # make X-AdaCBM clearly different from AdaCBM
        x.attention_block.cross_attn.gate.fill_(0.3)
        x.attn_blend.fill_(0.2)
        x.asso_mat_bias.add_(torch.randn(C))      # own head != AdaCBM head
        model = x if which == "xadacbm" else ada
        out = model(torch.randn(32, D))
        logits, scores = out[0], out[1]
        redone = intervention.logits_from_scores(model, scores)
    assert torch.allclose(logits, redone, atol=1e-5)
    assert torch.allclose(torch.softmax(logits, -1), torch.softmax(redone, -1), atol=1e-6)


def test_using_the_wrong_models_head_would_be_caught(cfg, tmp_path):             # regression for issue 5
    ada = _randomised_adacbm(cfg)
    x = _x_from(ada, cfg, tmp_path)
    with torch.no_grad():
        x.asso_mat_bias.add_(torch.randn(C))
        logits, scores, _ = x(torch.randn(16, D))
        wrong = intervention.logits_from_scores(ada, scores)
    assert not torch.allclose(logits, wrong, atol=1e-3)


def test_concept_to_class_mapping_comes_from_metadata(cfg):                      # issue 9
    m = _randomised_adacbm(cfg)
    assert intervention.concept_classes(m) == CLS_OF.tolist()


def test_trained_checkpoint_roundtrip_and_strict_loading(cfg, tmp_path):         # issues 2 + 4
    ada = _randomised_adacbm(cfg)
    x = _x_from(ada, cfg, tmp_path)
    with torch.no_grad():
        x.attention_block.cross_attn.gate.fill_(0.4)          # "learned" values
        x.attn_blend.fill_(-0.1)
    ck = tmp_path / "best.ckpt"
    torch.save({"state_dict": x.state_dict(), "hyper_parameters": {"cfg": dict(cfg)}}, ck)
    loaded = model_io.load_xadacbm(ck, tmp_path, "cpu")
    feats = torch.randn(16, D)
    with torch.no_grad():
        assert torch.allclose(x(feats)[0], loaded(feats)[0], atol=1e-6)
    assert torch.tanh(loaded.attention_block.cross_attn.gate).item() == pytest.approx(np.tanh(0.4), abs=1e-6)
    bad = {k: v for k, v in x.state_dict().items() if k != "asso_mat_bias"}
    torch.save({"state_dict": bad, "hyper_parameters": {"cfg": dict(cfg)}}, ck)
    with pytest.raises(RuntimeError):
        model_io.load_xadacbm(ck, tmp_path, "cpu")


def test_adacbm_to_xadacbm_transfer_rejects_missing_tensors(cfg, tmp_path):
    ada = _randomised_adacbm(cfg)
    sd = {k: v for k, v in ada.state_dict().items() if k != "asso_mat"}
    p = tmp_path / "broken.ckpt"
    torch.save({"state_dict": sd}, p)
    with pytest.raises(RuntimeError):
        XAdaCBMConcept(Config(dict(cfg))).load_from_adacbm_checkpoint(p)
