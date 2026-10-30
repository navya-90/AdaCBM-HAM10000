#!/usr/bin/env bash
# Re-evaluate a trained AdaCBM run on the TEST split with the Lightning trainer.
# Usage: bash test.sh [run_name]
set -e
RUN_NAME=${1:-adacbm_ham10000}
RESULTS=${RESULTS_DIR:-./results}
python main.py --cfg cfg/HAM10000/HAM10000_allshot_fac.py --test --run-name "${RUN_NAME}" \
  --func our_main --cfg-options \
  "ckpt_path='${RESULTS}/runs/${RUN_NAME}/checkpoints/last.ckpt'" \
  "concept_select_fn='our_selection'" "num_concept=70" "clip_model='ViT-L/14'" \
  "use_img_norm=False" "use_txt_norm=False" "use_residual=False" "bs=256" \
  "num_layers=2" "pearson_weight=0.9" "freeze_backbone=True"
