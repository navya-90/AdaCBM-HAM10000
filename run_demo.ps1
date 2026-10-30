# Launch the dashboard. It serves the TRAINED checkpoints under .\results\runs\ (copy them
# from Kaggle first) -- or set ADACBM_CKPT / XADACBM_CKPT / CONCEPTS_DIR. Images: set HAM_IMAGE_ROOT.
Write-Host "Dashboard: http://127.0.0.1:8000  (Ctrl+C to stop)"
Start-Process "http://127.0.0.1:8000"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
