# PNAC Code

This folder contains a minimal, reproducible implementation of **Pseudolabel Amplification Cascade (PNAC)**.
It trains a baseline classifier on the labeled split, iteratively pseudo-labels the unlabeled pool
with high-confidence predictions (using TTA), and tracks F1 decay on the validation set.

## File Structure

- `pnac.py` - Core PNAC diagnostic implementation
- `run_calibration.py` - Automation script to run experiments for all noise levels
- `plot_pnac_results.py` - Generate publication figures from results
- `generate_synthetic_plots.py` - Generate placeholder synthetic plots
- `viz_dataset.py` - Visualize dataset samples

## Quick Start: Full Calibration Workflow

### Step 1: Run Calibration Experiments

Run PNAC for all noise levels (0%, 10%, 20%, 30%, 40%, 50%):

```bash
python run_calibration.py
```

This will:
- Run PNAC experiments for each noise level
- Save results to `pnac_results/noise_X.X/`
- Create summary.json files with decay rates (beta)

**Note:** This takes approximately 30-60 minutes depending on GPU.

### Step 2: Generate Figures

After experiments complete, generate publication figures:

```bash
python plot_pnac_results.py
```

This creates:
- `../figures/decay_curves.pdf` - F1 decay curves for all noise rates
- `../figures/beta_vs_noise.pdf` - Calibration curve (beta vs noise)

## Single Experiment

Run a single PNAC experiment:

```bash
python pnac.py \
  --data-root "../../code 04112025/three_classes_dataset" \
  --manifest "../../code 04112025/three_classes_split_manifest.json" \
  --noise-rate 0.3 \
  --iterations 6 \
  --epochs 5 \
  --confidence 0.95 \
  --tta 10 \
  --pretrained \
  --output-dir runs
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-root` | required | Path to dataset root directory |
| `--manifest` | required | Path to split manifest JSON |
| `--output-dir` | `runs` | Output folder for results |
| `--noise-rate` | 0.0 | Fraction of labels to corrupt (0.0-0.5) |
| `--iterations` | 6 | Number of PNAC iterations (manuscript: 5) |
| `--epochs` | 5 | Training epochs per iteration |
| `--batch-size` | 32 | Training batch size (manuscript: 160) |
| `--confidence` | 0.95 | Confidence threshold for pseudo-labels (manuscript: 0.85) |
| `--tta` | 10 | Number of TTA runs per image (manuscript: 3) |
| `--lr` | 0.001 | Learning rate |
| `--seed` | 42 | Random seed |
| `--pretrained` | False | Use pretrained ResNet weights (manuscript: True) |

> **Note:** The defaults above are for general use. The manuscript experiments use `run_calibration.py`, which overrides several defaults: `--iterations 5 --confidence 0.85 --tta 3 --batch-size 160 --pretrained`.

## Outputs

Each run creates a folder containing:
- `metrics.csv` - Iteration-wise validation accuracy and F1
- `summary.json` - Decay rate (beta) and configuration
- `pseudo_labels/iter_XX.json` - Selected pseudo-labels per iteration

## Requirements

See `requirements.txt` in this folder.

```
torch>=2.0
torchvision>=0.15
numpy
Pillow
matplotlib
```
