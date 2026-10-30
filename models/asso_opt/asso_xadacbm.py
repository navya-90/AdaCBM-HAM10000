"""
X-AdaCBM: Cross-Modal Concept-Guided Attention Adaptive Concept Bottleneck Model.
Novel architectural extension to AdaCBM (MICCAI 2024).

Key Innovation:
Instead of transforming image embeddings through an isolated static MLP (which is blind to 
what concepts are being tested), X-AdaCBM incorporates a Multi-Head Cross-Attention Adapter 
(CACL - Concept-Aware Cross Layer) where visual embeddings dynamically query the semantic 
concept dictionary. This bridges the medical domain gap conditionally based on active 
pathological features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from pathlib import Path
import os

from models.asso_opt.asso_our import AssoConcept

device = "cuda" if torch.cuda.is_available() else "cpu"


class CrossModalConceptAttention(nn.Module):
    """
    Cross-Attention mechanism where image queries the text concept dictionary.
    Allows image features to be refined conditioned on the semantic concept space.
    """
    def __init__(self, dim, num_heads=4, dropout=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        # Learnable gating factor initialized to 0:
        # Ensures smooth residual departure from the base AdaCBM solution.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, img_feat, concept_feat):
        """
        img_feat: Tensor of shape [B, D]
        concept_feat: Tensor of shape [K, D] or [B, K, D]
        """
        q = img_feat.unsqueeze(1) if img_feat.dim() == 2 else img_feat  # [B, 1, D]
        
        if concept_feat.dim() == 2:
            k = concept_feat.unsqueeze(0).expand(q.shape[0], -1, -1)     # [B, K, D]
            v = k
        else:
            k = concept_feat
            v = concept_feat

        attn_out, attn_weights = self.multihead_attn(q, k, v)           # [B, 1, D], [B, 1, K]
        
        # Gated residual connections. BOTH the attention and the feed-forward residual are
        # scaled by tanh(gate), so gate == 0 gives h == q exactly and the block is an identity
        # (X-AdaCBM then reproduces AdaCBM; verified in tests/test_consistency.py).
        g = torch.tanh(self.gate)
        h = q + g * self.norm1(attn_out)
        h = h + g * self.ffn(self.norm2(h))
        return h.squeeze(1), attn_weights.squeeze(1)


class XOurModule(nn.Module):
    """
    Dual-branch Adaptive Module:
    1. Base MLP adapter (matching AdaCBM parameterization for backward compatibility)
    2. Cross-modal concept-guided attention module
    """
    def __init__(self, dim, num_layers=2, num_heads=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.LeakyReLU(0.01))
        self.linear1 = nn.Sequential(*layers)
        self.cross_attn = CrossModalConceptAttention(dim=dim, num_heads=num_heads)

    def forward(self, original_emb, concept_feat=None, residual=False, use_img_norm=False):
        # 1. Base non-linear feature projection
        A = self.linear1(original_emb)
        if use_img_norm:
            A = A / (A.norm(dim=-1, keepdim=True) + 1e-7)
        if residual:
            A = A + original_emb

        attn_weights = None
        # 2. Dynamic concept cross-attention
        if concept_feat is not None:
            A, attn_weights = self.cross_attn(A, concept_feat)
            if use_img_norm:
                A = A / (A.norm(dim=-1, keepdim=True) + 1e-7)

        return A, attn_weights


class XAdaCBMConcept(AssoConcept):
    """
    X-AdaCBM LightningModule incorporating cross-modal concept-guided attention.
    """
    def __init__(self, cfg, init_weight=None, select_idx=None) -> None:
        super().__init__(cfg, init_weight, select_idx)
        dim = self.concepts.shape[-1] if hasattr(self, 'concepts') and self.concepts is not None else 768
        num_heads = getattr(cfg, 'num_heads', 4)
        num_layers = getattr(cfg, 'num_layers', 2)

        self.attention_block = XOurModule(dim=dim, num_layers=num_layers, num_heads=num_heads)
        self.cfg.asso_act = ''
        self.mask = self.init_weight.clone().to(device)
        self.asso_mat = torch.nn.Parameter(self.init_weight.clone())
        self.dot_product_bias = torch.nn.Parameter(torch.zeros(len(self.select_idx)))
        self.asso_mat_bias = torch.nn.Parameter(torch.zeros(self.cfg.num_cls))
        self.attn_blend = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image):
        mat = self._get_weight_mat()
        # Cross-modal conditioning on concept embeddings
        image_embed, attn_weights = self.attention_block(
            image,
            concept_feat=self.concepts,
            residual=self.cfg.use_residual,
            use_img_norm=self.cfg.use_img_norm
        )
        # Concept bottleneck score
        dot_product = image_embed @ self.concepts.transpose(-2, -1)
        dot_product = dot_product + self.dot_product_bias

        # Optional soft saliency injection from attention weights
        if attn_weights is not None:
            dot_product = dot_product + torch.tanh(self.attn_blend) * attn_weights

        # Diagnosis logits
        pred = dot_product @ mat.t()
        pred = pred + self.asso_mat_bias
        return pred, dot_product, attn_weights

    def configure_optimizers(self):
        """AdamW, used by train_xadacbm.py (and by Lightning if you train this module with a
        Trainer). Fine-tuning from AdaCBM uses a small LR; no scheduler."""
        wd = self.cfg.get('weight_decay', 1e-4) if hasattr(self.cfg, 'get') else 1e-4
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=wd)

    def _get_weight_mat(self):
        mat = self.asso_mat
        mat = mat * self.mask
        return mat

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        pred, _, _ = self.forward(image)
        cls_loss = F.cross_entropy(pred, label)

        self.log('training_loss', cls_loss)
        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return cls_loss

    def validation_step(self, batch, batch_idx):
        image, y = batch
        pred, _, _ = self.forward(image)
        loss = F.cross_entropy(pred, y)
        self.log('val_loss', loss)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        pred, _, _ = self.forward(image)
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def load_from_adacbm_checkpoint(self, ckpt_path):
        """Copy the AdaCBM weights (adapter MLP, association matrix, biases) into X-AdaCBM.
        Fails loudly if any AdaCBM tensor is missing or has a different shape. The new
        cross-attention parameters keep their initialisation (gate == 0, attn_blend == 0)."""
        import compat  # noqa: F401  (registers the mmcv stand-in used inside the checkpoint)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src = ckpt['state_dict']
        own = self.state_dict()
        new_only = lambda k: k.startswith('attention_block.cross_attn.') or k == 'attn_blend'
        problems = []
        for k, v in src.items():
            if k not in own:
                problems.append(f"unexpected key in AdaCBM checkpoint: {k}")
            elif own[k].shape != v.shape:
                problems.append(f"shape mismatch for {k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}")
        for k in own:
            if k not in src and not new_only(k):
                problems.append(f"AdaCBM tensor missing from checkpoint: {k}")
        if problems:
            raise RuntimeError("Cannot initialise X-AdaCBM from AdaCBM checkpoint:\n  " + "\n  ".join(problems))
        with torch.no_grad():
            for k, v in src.items():
                own[k].copy_(v)
        print(f"Loaded {len(src)} AdaCBM tensors into X-AdaCBM (gate/attn_blend left at 0).")
        return self
