#!/usr/bin/env python3
"""
PNAC Calibration Runner
-----------------------
Automates running the PNAC diagnostic for multiple noise levels.
This script runs pnac.py for each noise rate in the calibration set
and collects results for subsequent plotting.
"""

import subprocess
import sys
import os
from pathlib import Path

# --- CONFIGURATION ---
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths relative to the project structure
# The dataset is in the local data/ directory
DATA_ROOT = (PROJECT_ROOT / "data" / "three_classes_dataset").resolve()
MANIFEST = (PROJECT_ROOT / "data" / "three_classes_split_manifest.json").resolve()
OUTPUT_DIR = (SCRIPT_DIR / "pnac_results").resolve()

# Python executable (uses the current environment)
PYTHON_EXE = sys.executable

# PNAC script path
PNAC_SCRIPT = SCRIPT_DIR / "pnac.py"


def is_experiment_complete(noise_rate: float, seed: int) -> bool:
    """Check if an experiment has already completed (has summary.json)."""
    run_output = OUTPUT_DIR / f"noise_{noise_rate:.1f}" / f"seed_{seed}"
    summary_files = list(run_output.glob("**/summary.json"))
    return len(summary_files) > 0


def run_experiment(noise_rate: float, epochs: int = 5, iterations: int = 6,
                   confidence: float = 0.95, tta: int = 10, seed: int = 42):
    """
    Run a single PNAC experiment for a given noise rate.

    Parameters:
        noise_rate: The fraction of labels to corrupt (0.0 to 0.5)
        epochs: Number of training epochs per iteration
        iterations: Number of PNAC iterations
        confidence: Confidence threshold for pseudo-label selection
        tta: Number of test-time augmentation runs
        seed: Random seed for reproducibility
    """
    # Check if already completed
    if is_experiment_complete(noise_rate, seed):
        print(f"\n[SKIP] Experiment already complete: noise={noise_rate:.1f}, seed={seed}")
        return True  # Return True to indicate success (already done)

    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: NOISE RATE {noise_rate:.1f}, SEED {seed}")
    print(f"{'='*60}\n")

    # Create output directory for this noise level and seed
    run_output = OUTPUT_DIR / f"noise_{noise_rate:.1f}" / f"seed_{seed}"
    run_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXE, str(PNAC_SCRIPT),
        "--data-root", str(DATA_ROOT),
        "--manifest", str(MANIFEST),
        "--output-dir", str(run_output),
        "--noise-rate", str(noise_rate),
        "--iterations", str(iterations),
        "--epochs", str(epochs),
        "--confidence", str(confidence),
        "--tta", str(tta),
        "--seed", str(seed),
        "--batch-size", "160",  # Optimized for RTX 3090 24GB
        "--num-workers", "2",   # Limited for Windows shared memory
        "--pretrained",  # Use pretrained weights for faster convergence
    ]

    print(f"Command: {' '.join(cmd[:4])} ...")
    print(f"Data root: {DATA_ROOT}")
    print(f"Manifest: {MANIFEST}")
    print()

    try:
        subprocess.check_call(cmd)
        print(f"\nExperiment completed successfully for noise rate {noise_rate}, seed {seed}")
    except subprocess.CalledProcessError as e:
        print(f"\nError running experiment for noise rate {noise_rate}, seed {seed}: {e}")
        raise


def verify_paths():
    """Verify that all required paths exist."""
    errors = []

    if not DATA_ROOT.exists():
        errors.append(f"Data root not found: {DATA_ROOT}")

    if not MANIFEST.exists():
        errors.append(f"Manifest file not found: {MANIFEST}")

    if not PNAC_SCRIPT.exists():
        errors.append(f"PNAC script not found: {PNAC_SCRIPT}")

    if errors:
        print("ERROR: Required files/directories not found:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease verify the paths in this script.")
        sys.exit(1)

    print("All paths verified successfully.")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Manifest: {MANIFEST}")
    print(f"  Output: {OUTPUT_DIR}")
    print()


def main():
    """Run the full PNAC calibration experiment."""
    print("PNAC Calibration Experiment Runner")
    print("="*60)
    print()

    # Verify paths first
    verify_paths()

    # Create output directory (keep existing results for resume)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count already completed experiments
    completed_count = sum(
        1 for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for seed in [42, 123, 456]
        if is_experiment_complete(rho, seed)
    )
    if completed_count > 0:
        print(f"Found {completed_count} completed experiments (will skip these)")
        print(f"To start fresh, run: python clear_results.py")

    # Noise levels to test (as described in the manuscript)
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]

    # Experiment parameters (optimized for speed and cascade effect)
    epochs = 5       # Epochs per iteration (pretrained weights converge fast)
    iterations = 5   # PNAC iterations (key insights visible in 5)
    confidence = 0.85  # Lower threshold to allow cascade with noisy data
    tta = 3          # TTA runs (3x faster than 10)
    seeds = [42, 123, 456]  # Multiple seeds for error bars

    print(f"Experiment configuration:")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Seeds: {seeds}")
    print(f"  Epochs per iteration: {epochs}")
    print(f"  PNAC iterations: {iterations}")
    print(f"  Confidence threshold: {confidence}")
    print(f"  TTA runs: {tta}")
    print(f"  Batch size: 160")
    print(f"  Data workers: 2")
    print(f"  NOTE: Using faster settings (epochs=5, iter=5, TTA=3, conf=0.85)")
    print(f"  Total runs: {len(noise_levels)} x {len(seeds)} = {len(noise_levels) * len(seeds)}")
    print()

    # Run experiments for each noise level and seed
    successful = []
    skipped = []
    failed = []

    for rho in noise_levels:
        for seed in seeds:
            try:
                # Check if already complete before running
                if is_experiment_complete(rho, seed):
                    skipped.append((rho, seed))
                    print(f"[SKIP] Already complete: noise={rho:.1f}, seed={seed}")
                    continue

                run_experiment(
                    noise_rate=rho,
                    epochs=epochs,
                    iterations=iterations,
                    confidence=confidence,
                    tta=tta,
                    seed=seed,
                )
                successful.append((rho, seed))
            except Exception as e:
                print(f"Failed for noise rate {rho}, seed {seed}: {e}")
                failed.append((rho, seed))
                # Continue with other experiments
                continue

    # Summary
    total = len(noise_levels) * len(seeds)
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"\nTotal experiments: {total}")
    print(f"  Newly completed: {len(successful)}")
    print(f"  Previously completed (skipped): {len(skipped)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\nFailed experiments:")
        for rho, seed in failed:
            print(f"    - noise={rho:.1f}, seed={seed}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nNext step: Run plot_pnac_results.py to generate figures.")
    print("To clear all results and start fresh: python clear_results.py")


def evaluate_test_set():
    """Evaluate saved checkpoints on the held-out test set.

    Loads iter_00 (t=0) and iter_05 (t=T) checkpoints for four key noise
    levels across three seeds and reports test-set macro-F1.
    """
    import json
    import csv
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    # Import model/data utilities from pnac.py (same directory)
    sys.path.insert(0, str(SCRIPT_DIR))
    from pnac import (DatasetConfig, ManifestDataset, load_manifest,
                      build_model, build_transforms, evaluate)

    cfg = DatasetConfig(data_root=DATA_ROOT, manifest_path=MANIFEST)
    class_names, _, _, _, test_items, _ = load_manifest(cfg.manifest_path)
    _, eval_tf, _ = build_transforms(cfg)
    num_classes = len(class_names)

    test_dataset = ManifestDataset(DATA_ROOT, test_items, transform=eval_tf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    test_loader = DataLoader(
        test_dataset, batch_size=160, shuffle=False,
        num_workers=2, pin_memory=use_cuda
    )

    print(f"Test-set evaluation: {len(test_items)} samples, device: {device}")

    noise_levels = [0.0, 0.3, 0.45, 0.60]
    seeds = [42, 123, 456]
    iters_to_eval = [0, 5]  # t=0 and t=T

    results = []
    for noise in noise_levels:
        for seed in seeds:
            # Find directory (handle varying decimal formatting on disk)
            seed_dir = None
            for fmt in [f"noise_{noise}", f"noise_{noise:.1f}", f"noise_{noise:.2f}"]:
                candidate = OUTPUT_DIR / fmt / f"seed_{seed}"
                if candidate.exists():
                    seed_dir = candidate
                    break
            if seed_dir is None:
                print(f"  [SKIP] No directory for noise={noise}, seed={seed}")
                continue

            # Use the most recent run
            run_dirs = sorted(seed_dir.glob("pnac_*"), key=lambda p: p.name,
                              reverse=True)
            if not run_dirs:
                print(f"  [SKIP] No runs in {seed_dir}")
                continue
            run_dir = run_dirs[0]

            # Also read validation F1 from summary.json for comparison
            summary_path = run_dir / "summary.json"
            val_f1_by_iter = {}
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                for i, f1 in enumerate(summary.get("f1_scores", [])):
                    val_f1_by_iter[i] = f1

            for it in iters_to_eval:
                ckpt_path = run_dir / "models" / f"model_iter_{it:02d}.pt"
                if not ckpt_path.exists():
                    print(f"  [SKIP] Missing {ckpt_path.name} "
                          f"for noise={noise}, seed={seed}")
                    continue

                ckpt = torch.load(ckpt_path, map_location=device,
                                  weights_only=False)
                model = build_model(num_classes, pretrained=False).to(device)
                model.load_state_dict(ckpt["model_state_dict"])

                acc, f1, _ = evaluate(model, test_loader, device, num_classes)
                val_f1 = val_f1_by_iter.get(it)

                results.append({
                    "noise": noise, "seed": seed, "iteration": it,
                    "test_f1": round(f1, 4), "test_acc": round(acc, 4),
                    "val_f1": round(val_f1, 4) if val_f1 is not None else None,
                })
                vstr = f"{val_f1:.4f}" if val_f1 is not None else "N/A"
                print(f"  noise={noise:.2f} seed={seed} t={it}: "
                      f"test_F1={f1:.4f}  val_F1={vstr}")

    # Aggregate: mean +/- std across seeds
    print(f"\n{'='*70}")
    print(f"{'Noise':>6}  {'Iter':>4}  {'Test F1 (mean +/- std)':>22}  "
          f"{'Val F1 (mean +/- std)':>22}")
    print(f"{'-'*70}")
    for noise in noise_levels:
        for it in iters_to_eval:
            subset = [r for r in results
                      if r["noise"] == noise and r["iteration"] == it]
            if not subset:
                continue
            tf1 = np.array([r["test_f1"] for r in subset])
            vf1 = np.array([r["val_f1"] for r in subset
                            if r["val_f1"] is not None])
            t_str = f"{tf1.mean():.4f} +/- {tf1.std():.4f}"
            v_str = (f"{vf1.mean():.4f} +/- {vf1.std():.4f}"
                     if len(vf1) > 0 else "N/A")
            print(f"{noise:>6.2f}  {it:>4d}  {t_str:>22s}  {v_str:>22s}")
    print(f"{'='*70}")

    # Save per-seed results to CSV
    csv_path = OUTPUT_DIR / "test_set_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["noise", "seed", "iteration",
                           "test_f1", "test_acc", "val_f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval-test":
        evaluate_test_set()
    else:
        main()
