# PNAC - Pseudo-label Noise Amplification Cascade

## Quick Start

| What you want to do | Run this |
|---------------------|----------|
| **First time setup** | `SETUP_CONDA.bat` |
| **Run ALL experiments (transition + confidence)** | `RUN_FULL_STUDY.bat` |
| **Generate manuscript figures** | `GENERATE_MANUSCRIPT_FIGURES.bat` |
| **Analyze results** | `ANALYZE_STUDIES.bat` |

## One-Time Setup

| BAT File | What it does |
|----------|-------------|
| `SETUP_CONDA.bat` | Creates the `pnac` conda environment (Python 3.11, PyTorch + CUDA 12.1, dependencies). Run once before any experiments. |

## Experiment Runners

### Main Orchestrators

| BAT File | Experiments | Runtime | Description |
|----------|------------|---------|-------------|
| `RUN_FULL_STUDY.bat` | 69 | ~6-8 hours | **Recommended.** Runs both Part 1 (transition) and Part 2 (confidence study). Has skip logic - safe to restart. |
| `RUN_ALL_EXPERIMENTS.bat` | 18 | ~2-3 hours | Original baseline study: noise 0.0-0.5, confidence 0.85, 3 seeds each. |
| `RUN_CONFIDENCE_STUDY.bat` | 60 | ~5-6 hours | Confidence threshold study only: 5 noise levels x 4 confidence thresholds x 3 seeds. |
| `RUN_TRANSITION_STUDY.bat` | 9 | ~30-45 min | Transition zone only: noise 0.45, 0.55, 0.60, 3 seeds each. |

### Individual Noise Levels

Each runs 3 seeds at a single noise level with default confidence (0.85):

| BAT File | Noise Level |
|----------|------------|
| `RUN_NOISE_0.0.bat` | 0.0 (clean) |
| `RUN_NOISE_0.1.bat` | 0.1 |
| `RUN_NOISE_0.2.bat` | 0.2 |
| `RUN_NOISE_0.3.bat` | 0.3 |
| `RUN_NOISE_0.4.bat` | 0.4 |
| `RUN_NOISE_0.45.bat` | 0.45 |
| `RUN_NOISE_0.5.bat` | 0.5 |
| `RUN_NOISE_0.55.bat` | 0.55 |
| `RUN_NOISE_0.60.bat` | 0.60 |

### Core Scripts (called by orchestrators)

| BAT File | Description |
|----------|-------------|
| `run_noise_conda.bat` | Core runner. Usage: `run_noise_conda.bat <noise_level>`. Runs `pnac.py` for 3 seeds with confidence 0.85. Skips completed runs (checks for `summary.json`). |
| `run_noise_confidence_conda.bat` | Like above but with custom confidence. Usage: `run_noise_confidence_conda.bat <noise_level> <confidence>`. |

## Analysis and Visualization

| BAT File | Description |
|----------|-------------|
| `ANALYZE_STUDIES.bat` | Runs `analyze_studies.py` - analyzes all experiment results and prints summary tables. |
| `GENERATE_MANUSCRIPT_FIGURES.bat` | Generates publication-quality PDF figures for the manuscript (F1 trajectories, robustness horizon, failure modes, throughput vs noise). |
| `GENERATE_PLOTS.bat` | Generates general plots from both real and synthetic results. |
| `RUN_SYNTHETIC_ANALYSIS.bat` | Runs synthetic dataset experiments (noise sweep, difficulty sweep, mechanism validation). ~2-3 min. |

## Utilities

| BAT File | Description |
|----------|-------------|
| `CLEAR_RESULTS.bat` | **Destructive.** Deletes all experiment results and figures. |
| `cleanup_backups.bat` | Cleans up backup folders (dry run by default). |
| `start_auto_backup.bat` | Starts automatic backup of the project folder. |

## AI Assistant Launchers

| BAT File | Description |
|----------|-------------|
| `launch_claude.bat` | Opens Claude Code CLI in WSL. |
| `launch_codex.bat` | Opens Codex CLI in WSL. |
| `launch_gemini.bat` | Opens Gemini CLI in WSL. |
| `manuscript_review_loop.bat` | Interactive manuscript review loop with multiple AI model options. |
| `manuscript_review_loop_viewer.bat` | GUI viewer for manuscript review loop results. |

## Other

| BAT File | Description |
|----------|-------------|
| `RUN_PUBLICATION_GPU.bat` | Runs experiments via WSL with GPU support (alternative to direct Windows execution). |

## Resume / Restart Behavior

All experiment runners have **built-in skip logic**. Each completed experiment saves a `summary.json` file. When you re-run any BAT file, it checks for this file and skips already-completed experiments. This means:

- You can safely stop and restart at any time
- Only the currently-running experiment will need to re-run
- Completed experiments will show `[SKIP]` in the output

## Results Directory Structure

```
code/
  pnac_results/                          # Default confidence (0.85)
    noise_0.0/seed_42/pnac_YYYYMMDD_HHMMSS/
      metrics.csv                        # Per-iteration metrics
      summary.json                       # Final results
      models/                            # Saved model checkpoints
      pseudo_labels/                     # Pseudo-label data
    noise_0.0/seed_123/...
    ...
  pnac_results_confidence_study/         # Variable confidence
    conf_0.70/noise_0.0/seed_42/...
    conf_0.80/noise_0.0/seed_42/...
    conf_0.90/...
    conf_0.95/...
```

## Experiment Configuration (all experiments)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 (ImageNet pretrained) |
| Epochs per iteration | 5 |
| Pseudo-labeling iterations | 5 |
| TTA runs | 3 |
| Batch size | 160 |
| Seeds | 42, 123, 456 |
| Dataset | 4,626 labeled / 990 validation / 35,654 unlabeled |
