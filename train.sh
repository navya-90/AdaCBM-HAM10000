#!/usr/bin/env bash
# Train the AdaCBM baseline (concept selection on TRAIN features, model selection on VAL).
# Prerequisite: python scripts/extract_features.py
# Usage: bash train.sh [run_name] [seed]
set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID
RUN_NAME=${1:-adacbm_ham10000}
SEED=${2:-777}

python main.py --cfg cfg/HAM10000/HAM10000_allshot_fac.py \
  --run-name "${RUN_NAME}" \
  --func our_main \
  --cfg-options \
  "max_epochs=300" "concept_select_fn='our_selection'" "seed=${SEED}" \
  "num_concept=70" "lr=0.0005" "clip_model='ViT-L/14'" \
  "use_img_norm=False" "use_txt_norm=False" "use_residual=False" \
  "bs=256" "num_layers=2" "pearson_weight=0.9" "freeze_backbone=True"
