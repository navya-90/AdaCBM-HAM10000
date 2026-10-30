# Train X-AdaCBM (train/val only) and evaluate on the test split.
# Requires: features extracted and an AdaCBM baseline run under .\results\runs\ (see README).
python train_xadacbm.py --epochs 50 --lr 0.001
python evaluate_models.py --split test
