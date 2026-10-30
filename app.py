"""
FastAPI Server for X-AdaCBM Interactive Explainable AI Medical Diagnostic Dashboard.
Supports:
1. Real dermatoscopic image inference via CLIP ViT-L/14
2. Comparative prediction between AdaCBM (MICCAI 2024) and Proposed X-AdaCBM
3. Detailed Concept Bottleneck attribution (scores & cross-modal attention weights)
4. Interactive Doctor-in-the-Loop Concept Intervention with live counterfactual recalculation
"""

import os
import io
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

import compat  # noqa: F401
import intervention
import model_io
import project_paths as pp
from clip_loader import load_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
STATIC_DIR = str(pp.REPO_ROOT / "static")
app = FastAPI(title="X-AdaCBM Medical Diagnostic Dashboard", version="1.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------------------
# Model locations. No silent fallbacks: the dashboard serves the SAME checkpoints that were
# evaluated (see evaluate_models.py). Configure with environment variables:
#   RESULTS_DIR, BASELINE_RUN (default adacbm_ham10000), XADACBM_RUN (default xadacbm_ham10000)
#   or explicit ADACBM_CKPT / XADACBM_CKPT / CONCEPTS_DIR.
# ---------------------------------------------------------------------------------------
_root = pp.results_root()
_base_run = pp.run_dir(_root, os.environ.get("BASELINE_RUN", "adacbm_ham10000"))
_x_run = pp.run_dir(_root, os.environ.get("XADACBM_RUN", "xadacbm_ham10000"))
CONCEPTS_DIR = os.environ.get("CONCEPTS_DIR", str(_base_run / "concepts"))
ADACBM_CKPT = os.environ.get("ADACBM_CKPT", str(_base_run / "checkpoints" / "last.ckpt"))
XADACBM_CKPT = os.environ.get("XADACBM_CKPT", str(_x_run / "checkpoints" / "best.ckpt"))
for _p in (ADACBM_CKPT, XADACBM_CKPT, CONCEPTS_DIR):
    if not os.path.exists(_p):
        raise FileNotFoundError(
            f"{_p} not found. Copy the trained run folders from Kaggle into {_root}/runs/ "
            "or set ADACBM_CKPT / XADACBM_CKPT / CONCEPTS_DIR.")

print("Loading CLIP and trained models for the dashboard...")
clip_model, clip_preprocess = load_clip("ViT-L/14", device)
adacbm = model_io.load_adacbm(ADACBM_CKPT, CONCEPTS_DIR, device)
xadacbm = model_io.load_xadacbm(XADACBM_CKPT, CONCEPTS_DIR, device)   # trained weights, nothing overridden

_splits = pp.load_splits()
cls_names = pp.class_names_from_splits(_splits)
_img_root = pp.find_image_root()
IMAGE_INDEX = pp.build_image_index(_img_root) if _img_root else {}

concept_names = [str(c) for c in adacbm.concept_raw]
NUM_CONCEPTS = len(concept_names)
# class of each selected concept, from the saved concept metadata (not "index // 10")
CONCEPT_CLASS = intervention.concept_classes(adacbm)
assert len(CONCEPT_CLASS) == NUM_CONCEPTS == xadacbm.asso_mat.shape[1]


class InterventionRequest(BaseModel):
    concept_scores: List[float]
    model: str = "xadacbm"   # "xadacbm" | "adacbm"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/samples")
async def get_sample_images():
    """Sample images from the held-out TEST split (two per class) that exist locally."""
    samples = []
    for c in cls_names:
        present = [n for n in _splits["test"][c] if n in IMAGE_INDEX][:2]
        samples += [{"filename": n, "url": f"/api/image/{n}"} for n in present]
    return samples


@app.get("/api/image/{filename}")
async def get_sample_image(filename: str):
    p = IMAGE_INDEX.get(filename)   # dictionary lookup: no path traversal possible
    if p is None:
        raise HTTPException(status_code=404, detail="Image not found")
    with open(p, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


def process_image(pil_img):
    img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor).to(device)
    
    # Run AdaCBM
    with torch.no_grad():
        ada_logits, ada_cpt = adacbm(img_feat)
        ada_probs = torch.softmax(ada_logits, dim=-1).cpu().numpy()[0]
        ada_cpt_scores = ada_cpt.cpu().numpy()[0]

    # Run X-AdaCBM
    with torch.no_grad():
        x_logits, x_cpt, attn = xadacbm(img_feat)
        x_probs = torch.softmax(x_logits, dim=-1).cpu().numpy()[0]
        x_cpt_scores = x_cpt.cpu().numpy()[0]
        attn_scores = attn.cpu().numpy()[0] if attn is not None else np.zeros(NUM_CONCEPTS)

    # Format clinical concepts
    concepts_data = []
    for i in range(NUM_CONCEPTS):
        target_class_idx = CONCEPT_CLASS[i]
        concepts_data.append({
            "index": i,
            "name": str(concept_names[i]),
            "associated_class": cls_names[target_class_idx],
            "score_adacbm": float(ada_cpt_scores[i]),
            "score_xadacbm": float(x_cpt_scores[i]),
            "cross_attention": float(attn_scores[i])
        })

    # Sort by active concept score descending
    sorted_concepts = sorted(concepts_data, key=lambda c: abs(c["score_xadacbm"]), reverse=True)

    return {
        "classes": cls_names,
        "adacbm": {
            "probabilities": [float(p) for p in ada_probs],
            "predicted_class": cls_names[int(np.argmax(ada_probs))],
            "confidence": float(np.max(ada_probs) * 100.0)
        },
        "xadacbm": {
            "probabilities": [float(p) for p in x_probs],
            "predicted_class": cls_names[int(np.argmax(x_probs))],
            "confidence": float(np.max(x_probs) * 100.0)
        },
        "concepts": concepts_data,
        "top_active_concepts": sorted_concepts[:12]
    }


@app.post("/api/predict_sample")
async def predict_sample(filename: str = Form(...)):
    p = IMAGE_INDEX.get(filename)
    if p is None:
        raise HTTPException(status_code=404, detail="Sample image not found")
    img = Image.open(p).convert("RGB")
    res = process_image(img)
    res["sample_filename"] = filename
    return res


@app.post("/api/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    res = process_image(img)
    res["sample_filename"] = file.filename
    return res


@app.post("/api/intervene")
async def doctor_intervention(req: InterventionRequest):
    """
    Doctor-in-the-loop recalculation through the classification layer of the SAME model that
    produced the concept scores (association matrix, mask and class bias of that model):
        logits = scores @ (W * M)^T + b_cls
    Unchanged scores therefore reproduce the model's own logits (see tests/test_consistency.py).
    """
    if req.model not in ("xadacbm", "adacbm"):
        raise HTTPException(status_code=400, detail="model must be 'xadacbm' or 'adacbm'")
    if len(req.concept_scores) != NUM_CONCEPTS:
        raise HTTPException(status_code=400, detail=f"Must provide exactly {NUM_CONCEPTS} concept scores")
    model = xadacbm if req.model == "xadacbm" else adacbm
    with torch.no_grad():
        logits = intervention.logits_from_scores(model, req.concept_scores)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "probabilities": [float(p) for p in probs],
        "predicted_class": cls_names[pred_idx],
        "confidence": float(probs[pred_idx] * 100.0),
        "logits": [float(l) for l in logits.cpu().numpy()]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
