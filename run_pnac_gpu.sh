#!/bin/bash
# ============================================================================
# PNAC Publication Experiments with GPU (WSL)
# ============================================================================
# This script runs in WSL with CUDA GPU support
# To stop: Press Ctrl+C
# ============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================================"
echo "       PNAC PUBLICATION EXPERIMENTS (GPU via WSL)"
echo "============================================================================"
echo ""
echo "Working directory: $PWD"
echo ""

# Check if venv exists
if [ ! -d "venv_wsl" ]; then
    echo "[0/5] Creating virtual environment..."
    python3 -m venv venv_wsl
fi

echo "[1/5] Activating virtual environment..."
source venv_wsl/bin/activate

echo "[2/5] Installing dependencies with CUDA GPU support..."
pip install --upgrade pip -q
pip install numpy pillow matplotlib scipy tqdm -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo ""
echo "Verifying PyTorch + CUDA installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

cd code

echo ""
echo "[3/5] Running PNAC Calibration (6 noise levels)..."
echo ""
python run_calibration.py

echo ""
echo "[4/5] Generating plots from real results..."
echo ""
python plot_pnac_results.py

echo ""
echo "[5/5] Generating synthetic visualizations..."
echo ""
python generate_synthetic_plots.py

echo ""
echo "============================================================================"
echo "DONE! Results in: code/pnac_results/ and figures/"
echo "============================================================================"
