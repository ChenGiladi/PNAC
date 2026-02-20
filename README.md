# PNAC — Pseudo-Label Noise Amplification Cascade

A diagnostic framework that reveals dataset robustness to label noise through iterative pseudo-labeling. Accompanies the manuscript *"Diagnosing Label Noise Robustness via Pseudo-Label Noise Amplification Cascade"* (submitted to MDPI *Journal of Imaging*).

## Repository Structure

```
code/
  pnac.py                        # Core PNAC implementation
  run_calibration.py              # Run experiments across noise levels and seeds
  analyze_studies.py              # Analyze results and print summary tables
  generate_manuscript_figures.py  # Generate publication figures
  synthetic_pnac_analysis.py      # Synthetic dataset validation experiments
  requirements.txt                # Python dependencies
data/
  three_classes_split_manifest.json   # Train/val/unlabeled split definition
```

## Setup

```bash
# Create environment (Python 3.11, CUDA 12.1)
conda create -n pnac python=3.11 -y
conda activate pnac
pip install -r code/requirements.txt
```

## Usage

### Single Experiment

Run one PNAC experiment at a specific noise rate and seed:

```bash
python code/pnac.py \
  --data-root data/three_classes_dataset \
  --manifest data/three_classes_split_manifest.json \
  --noise-rate 0.3 \
  --iterations 5 \
  --epochs 5 \
  --confidence 0.85 \
  --tta 3 \
  --batch-size 160 \
  --pretrained \
  --seed 42 \
  --output-dir code/pnac_results/noise_0.3/seed_42
```

### Main Study (9 noise rates x 3 seeds = 27 experiments)

```bash
python code/run_calibration.py
```

This runs PNAC for noise rates 0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6 with seeds 42, 123, 456. Results are saved under `code/pnac_results/`. Completed runs (detected via `summary.json`) are skipped on restart.

### Confidence Threshold Study (5 noise rates x 4 thresholds x 3 seeds = 60 experiments)

```bash
for tau in 0.70 0.80 0.90 0.95; do
  for rho in 0.0 0.2 0.4 0.5 0.6; do
    for seed in 42 123 456; do
      python code/pnac.py \
        --data-root data/three_classes_dataset \
        --manifest data/three_classes_split_manifest.json \
        --noise-rate $rho \
        --confidence $tau \
        --iterations 5 --epochs 5 --tta 3 --batch-size 160 --pretrained \
        --seed $seed \
        --output-dir "code/pnac_results_confidence_study/conf_${tau}/noise_${rho}/seed_${seed}"
    done
  done
done
```

### Synthetic Validation

```bash
python code/synthetic_pnac_analysis.py
```

### Analysis and Figures

```bash
python code/analyze_studies.py
python code/generate_manuscript_figures.py
```

## Command-Line Options

| Option | Default | Manuscript value | Description |
|--------|---------|-----------------|-------------|
| `--data-root` | *(required)* | `data/three_classes_dataset` | Path to image dataset |
| `--manifest` | *(required)* | `data/three_classes_split_manifest.json` | Train/val/unlabeled split |
| `--noise-rate` | 0.0 | 0.0–0.6 | Fraction of labels to corrupt |
| `--iterations` | 6 | 5 | Pseudo-labeling rounds |
| `--epochs` | 5 | 5 | Training epochs per round |
| `--confidence` | 0.95 | 0.85 | Confidence threshold τ |
| `--tta` | 10 | 3 | Test-time augmentation runs |
| `--batch-size` | 32 | 160 | Training batch size |
| `--seed` | 42 | 42, 123, 456 | Random seed |
| `--pretrained` | False | True | Use ImageNet-pretrained ResNet-18 |

## Output Structure

Each experiment produces:

```
summary.json          # Final metrics, configuration, per-class F1
metrics.csv           # Per-iteration validation accuracy and F1
pseudo_labels/        # Selected pseudo-labels per iteration
models/               # Saved model checkpoints
```

## Dataset

The ultrasound phantom dataset (42,266 images, 3.9 GB) is not included in this repository due to size. Contact the corresponding author (chengi1@sce.ac.il) for access, or see the Data Availability section of the manuscript.

The split manifest (`data/three_classes_split_manifest.json`) defines the train (4,626) / validation (990) / unlabeled (35,654) partition used in all experiments.

## Reproducibility Checksums (SHA-256)

| File | SHA-256 (first 8) |
|------|-------------------|
| `code/pnac.py` | `9a90b44c` |
| `code/run_calibration.py` | `f56881cb` |
| `code/requirements.txt` | `5ada1a91` |
| `data/three_classes_split_manifest.json` | `350bc115` |

## License

See the manuscript for terms.
