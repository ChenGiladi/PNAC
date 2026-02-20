#!/bin/bash
# ============================================================================
# Run PNAC experiments for a specific noise level
# Usage: ./run_noise_level.sh <noise_level>
# Example: ./run_noise_level.sh 0.1
# ============================================================================

set -e

NOISE_LEVEL=$1

if [ -z "$NOISE_LEVEL" ]; then
    echo "Usage: $0 <noise_level>"
    echo "Example: $0 0.1"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================================"
echo "       PNAC EXPERIMENT - NOISE LEVEL $NOISE_LEVEL"
echo "============================================================================"
echo ""
echo "Working directory: $PWD"
echo ""

# Check if venv exists
if [ ! -d "venv_wsl" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_wsl
fi

echo "Activating virtual environment..."
source venv_wsl/bin/activate

# Check if PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install --upgrade pip -q
    pip install numpy pillow matplotlib scipy tqdm -q
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
fi

echo ""
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

cd code

# Configuration
DATA_ROOT="$SCRIPT_DIR/data/three_classes_dataset"
MANIFEST="$SCRIPT_DIR/data/three_classes_split_manifest.json"
OUTPUT_DIR="$SCRIPT_DIR/code/pnac_results"
SEEDS="42 123 456"
EPOCHS=10
ITERATIONS=8
CONFIDENCE=0.95
TTA=10
BATCH_SIZE=64
NUM_WORKERS=4

echo "Configuration:"
echo "  Noise level: $NOISE_LEVEL"
echo "  Seeds: $SEEDS"
echo "  Epochs: $EPOCHS"
echo "  Iterations: $ITERATIONS"
echo ""

for SEED in $SEEDS; do
    RUN_OUTPUT="$OUTPUT_DIR/noise_$NOISE_LEVEL/seed_$SEED"

    # Check if already completed
    if [ -f "$RUN_OUTPUT/summary.json" ]; then
        echo "[SKIP] Already complete: noise=$NOISE_LEVEL, seed=$SEED"
        continue
    fi

    echo "============================================================"
    echo "STARTING: noise=$NOISE_LEVEL, seed=$SEED"
    echo "============================================================"

    mkdir -p "$RUN_OUTPUT"

    python pnac.py \
        --data-root "$DATA_ROOT" \
        --manifest "$MANIFEST" \
        --output-dir "$RUN_OUTPUT" \
        --noise-rate "$NOISE_LEVEL" \
        --iterations "$ITERATIONS" \
        --epochs "$EPOCHS" \
        --confidence "$CONFIDENCE" \
        --tta "$TTA" \
        --seed "$SEED" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --pretrained

    echo ""
    echo "Completed: noise=$NOISE_LEVEL, seed=$SEED"
    echo ""
done

echo ""
echo "============================================================================"
echo "DONE! All experiments for noise level $NOISE_LEVEL completed."
echo "============================================================================"
