#!/usr/bin/env python3
"""
Generate publication-quality figures for the revised PNAC manuscript.

Figures generated:
1. f1_trajectories.pdf - F1 trajectories across iterations for different noise levels
2. robustness_horizon.pdf - Mean F1 vs noise level showing the transition
3. throughput_analysis.pdf - Pseudo-label throughput patterns
4. failure_modes.pdf - Comparison of starvation vs amplification at noise=0.55

Usage:
    python generate_manuscript_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from collections import defaultdict

# Configure matplotlib for publication-quality figures
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "pnac_results"
OUTPUT_DIR = SCRIPT_DIR.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_results():
    """Load all experimental results."""
    results = {}

    for noise_dir in sorted(RESULTS_DIR.glob("noise_*")):
        noise = noise_dir.name.replace("noise_", "")
        results[noise] = {}

        for seed_dir in noise_dir.glob("seed_*"):
            seed = seed_dir.name.replace("seed_", "")
            runs = sorted(seed_dir.iterdir(), key=lambda x: x.name, reverse=True)

            for run in runs:
                summary = run / "summary.json"
                metrics = run / "metrics.csv"

                if summary.exists():
                    with open(summary) as f:
                        data = json.load(f)

                    result = {
                        'f1_scores': data['f1_scores'],
                        'final_f1': data['f1_scores'][-1],
                        'initial_f1': data['f1_scores'][0],
                        'beta': data.get('decay_beta', None),
                    }

                    # Try to get pseudo-label counts from metrics
                    if metrics.exists():
                        import csv
                        with open(metrics) as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if rows and 'labeled_size' in rows[0]:
                                sizes = [int(r['labeled_size']) for r in rows]
                                pseudo_added = [sizes[i] - sizes[i-1] if i > 0 else 0
                                              for i in range(len(sizes))]
                                result['pseudo_labels'] = pseudo_added
                                result['total_pseudo'] = sum(pseudo_added)

                    results[noise][seed] = result
                    break

    return results


def figure1_f1_trajectories(results):
    """
    Figure 1: F1 trajectories across PNAC iterations for selected noise levels.
    Shows stable (low noise), transitional (moderate), and unstable (high noise) regimes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    iterations = np.arange(6)  # 0 to 5

    # Panel A: Stable regime (0.0, 0.2)
    ax = axes[0]
    colors_stable = ['#2E86AB', '#A23B72']
    for i, noise in enumerate(['0.0', '0.2']):
        if noise in results:
            for seed in ['42', '123', '456']:
                if seed in results[noise]:
                    f1s = results[noise][seed]['f1_scores']
                    ax.plot(iterations[:len(f1s)], f1s, 'o-',
                           color=colors_stable[i], alpha=0.6, markersize=5)
            # Plot mean
            all_f1s = [results[noise][s]['f1_scores'] for s in results[noise]]
            if all_f1s:
                mean_f1 = np.mean(all_f1s, axis=0)
                ax.plot(iterations[:len(mean_f1)], mean_f1, 's-',
                       color=colors_stable[i], linewidth=2.5, markersize=8,
                       label=f'ρ = {noise}')

    ax.set_xlabel('PNAC Iteration')
    ax.set_ylabel('Validation Macro-F1')
    ax.set_title('(A) Stable Regime', fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(-0.2, 5.2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5)

    # Panel B: Transition regime (0.4, 0.45)
    ax = axes[1]
    colors_trans = ['#F18F01', '#C73E1D']
    for i, noise in enumerate(['0.4', '0.45']):
        if noise in results:
            for seed in ['42', '123', '456']:
                if seed in results[noise]:
                    f1s = results[noise][seed]['f1_scores']
                    ax.plot(iterations[:len(f1s)], f1s, 'o-',
                           color=colors_trans[i], alpha=0.6, markersize=5)
            all_f1s = [results[noise][s]['f1_scores'] for s in results[noise]]
            if all_f1s:
                mean_f1 = np.mean(all_f1s, axis=0)
                ax.plot(iterations[:len(mean_f1)], mean_f1, 's-',
                       color=colors_trans[i], linewidth=2.5, markersize=8,
                       label=f'ρ = {noise}')

    ax.set_xlabel('PNAC Iteration')
    ax.set_title('(B) Transition Regime', fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(-0.2, 5.2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5)

    # Panel C: Unstable regime (0.5, 0.55, 0.6)
    ax = axes[2]
    colors_unstable = ['#6B2D5C', '#3D0C11', '#8B0000']
    for i, noise in enumerate(['0.5', '0.55', '0.6']):
        if noise not in results and noise == '0.6':
            noise = '0.60'  # Try alternate format
        if noise in results:
            for seed in ['42', '123', '456']:
                if seed in results[noise]:
                    f1s = results[noise][seed]['f1_scores']
                    ax.plot(iterations[:len(f1s)], f1s, 'o-',
                           color=colors_unstable[i], alpha=0.6, markersize=5)
            all_f1s = [results[noise][s]['f1_scores'] for s in results[noise]]
            if all_f1s:
                mean_f1 = np.mean(all_f1s, axis=0)
                label_noise = noise.rstrip('0').rstrip('.') if '.' in noise else noise
                ax.plot(iterations[:len(mean_f1)], mean_f1, 's-',
                       color=colors_unstable[i], linewidth=2.5, markersize=8,
                       label=f'ρ = {label_noise} (mean)')

    ax.set_xlabel('PNAC Iteration')
    ax.set_title('(C) Unstable / Chaos Regime', fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(-0.2, 5.2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'f1_trajectories.pdf')
    plt.savefig(OUTPUT_DIR / 'f1_trajectories.png')
    print(f"Saved: {OUTPUT_DIR / 'f1_trajectories.pdf'}")
    plt.close()


def figure2_robustness_horizon(results):
    """
    Figure 2: Mean F1 vs noise level showing the robustness horizon.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    noise_levels = sorted([float(n) for n in results.keys()])
    means = []
    stds = []

    for noise in noise_levels:
        noise_str = str(noise) if noise != int(noise) else f"{noise:.1f}"
        # Handle different string formats
        for fmt in [str(noise), f"{noise:.1f}", f"{noise:.2f}"]:
            if fmt in results:
                f1s = [results[fmt][s]['final_f1'] for s in results[fmt]]
                means.append(np.mean(f1s))
                stds.append(np.std(f1s))
                break
        else:
            means.append(np.nan)
            stds.append(np.nan)

    means = np.array(means)
    stds = np.array(stds)

    # Plot with error bars
    ax.errorbar(noise_levels, means, yerr=stds, fmt='o-', capsize=5, capthick=2,
                markersize=10, linewidth=2.5, color='#2E86AB', label='Mean F1 ± Std')

    # Highlight regions
    ax.axvspan(-0.02, 0.42, alpha=0.15, color='green', label='Stable (ρ ≤ 0.40)')
    ax.axvspan(0.42, 0.65, alpha=0.15, color='red', label='Transition (ρ ≥ 0.45)')

    # Threshold line
    ax.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5,
               label='F1 = 0.90 threshold')

    # Mark the robustness horizon
    ax.axvline(x=0.42, color='orange', linestyle=':', linewidth=2, alpha=0.8)
    ax.annotate('Robustness\nHorizon', xy=(0.42, 0.75), xytext=(0.32, 0.70),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='orange'))

    ax.set_xlabel('Noise Rate (ρ)', fontsize=12)
    ax.set_ylabel('Final Macro-F1', fontsize=12)
    ax.set_title('PNAC Robustness Horizon', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(-0.02, 0.65)
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'robustness_horizon.pdf')
    plt.savefig(OUTPUT_DIR / 'robustness_horizon.png')
    print(f"Saved: {OUTPUT_DIR / 'robustness_horizon.pdf'}")
    plt.close()


def figure3_failure_modes(results):
    """
    Figure 3: Comparison of starvation vs amplification at noise=0.55.
    Shows F1 trajectories and pseudo-label counts for the three seeds.
    """
    if '0.55' not in results:
        print("Warning: No data for noise=0.55, skipping failure modes figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    iterations = np.arange(6)
    colors = {'42': '#2E86AB', '123': '#A23B72', '456': '#F18F01'}
    markers = {'42': 'o', '123': 's', '456': '^'}

    # Panel A: F1 trajectories
    ax = axes[0]
    for seed in ['42', '123', '456']:
        if seed in results['0.55']:
            f1s = results['0.55'][seed]['f1_scores']
            total_pseudo = results['0.55'][seed].get('total_pseudo', 0)
            mode = 'Starvation' if total_pseudo == 0 else 'Mild cascade'
            ax.plot(iterations[:len(f1s)], f1s, f'{markers[seed]}-',
                   color=colors[seed], linewidth=2, markersize=8,
                   label=f'Seed {seed} ({mode})')

    ax.set_xlabel('PNAC Iteration', fontsize=11)
    ax.set_ylabel('Validation Macro-F1', fontsize=11)
    ax.set_title('(A) F1 Trajectories at ρ = 0.55', fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(-0.2, 5.2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5)

    # Panel B: Pseudo-label counts
    ax = axes[1]
    bar_width = 0.25
    x = np.arange(6)

    for i, seed in enumerate(['42', '123', '456']):
        if seed in results['0.55'] and 'pseudo_labels' in results['0.55'][seed]:
            pseudo = results['0.55'][seed]['pseudo_labels']
            # Pad to 6 elements if needed
            pseudo = pseudo + [0] * (6 - len(pseudo))
            ax.bar(x + i * bar_width, pseudo[:6], bar_width,
                  color=colors[seed], label=f'Seed {seed}', alpha=0.8)

    ax.set_xlabel('PNAC Iteration', fontsize=11)
    ax.set_ylabel('Pseudo-labels Added', fontsize=11)
    ax.set_title('(B) Pseudo-label Throughput at ρ = 0.55', fontweight='bold')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotations
    ax.annotate('Starvation\n(Seeds 42, 123)', xy=(1, 100), fontsize=9,
                ha='center', color='#2E86AB')
    ax.annotate('Mild cascade\n(Seed 456)', xy=(4, 800), fontsize=9,
                ha='center', color='#F18F01')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'failure_modes.pdf')
    plt.savefig(OUTPUT_DIR / 'failure_modes.png')
    print(f"Saved: {OUTPUT_DIR / 'failure_modes.pdf'}")
    plt.close()


def figure4_throughput_vs_noise(results):
    """
    Figure 4: Total pseudo-label throughput vs noise level.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    noise_data = []
    for noise in sorted(results.keys(), key=float):
        totals = []
        for seed in results[noise]:
            if 'total_pseudo' in results[noise][seed]:
                totals.append(results[noise][seed]['total_pseudo'])
        if totals:
            noise_data.append({
                'noise': float(noise),
                'mean': np.mean(totals),
                'std': np.std(totals),
                'values': totals
            })

    # Plot individual points
    for d in noise_data:
        for v in d['values']:
            ax.scatter(d['noise'], v, color='#2E86AB', alpha=0.4, s=50)

    # Exclude bimodal rho=0.55 from mean±std summary (per-seed only there)
    summary_data = [d for d in noise_data if d['noise'] != 0.55]
    sum_noises = [d['noise'] for d in summary_data]
    sum_means = [d['mean'] for d in summary_data]
    sum_stds = [d['std'] for d in summary_data]

    # Plot mean with error bars (excluding bimodal point)
    ax.errorbar(sum_noises, sum_means, yerr=sum_stds, fmt='s-', capsize=5, capthick=2,
                markersize=10, linewidth=2.5, color='#C73E1D',
                label='Mean ± Std (excluding ρ=0.55)', zorder=5)

    ax.set_xlabel('Noise Rate (ρ)', fontsize=12)
    ax.set_ylabel('Total Pseudo-labels Selected', fontsize=12)
    ax.set_title('Pseudo-label Throughput vs Noise Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.65)

    # Add annotation about throughput cliff
    ax.annotate('Throughput\ncliff', xy=(0.25, 9000),
               xytext=(0.15, 15000), fontsize=9,
               arrowprops=dict(arrowstyle='->', color='gray'))

    # Add annotation about bimodal behavior at high noise
    if any(d['noise'] == 0.55 for d in noise_data):
        ax.annotate('Bimodal\n(starvation vs\ncascade)', xy=(0.55, 3000),
                   xytext=(0.45, 8000), fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_vs_noise.pdf')
    plt.savefig(OUTPUT_DIR / 'throughput_vs_noise.png')
    print(f"Saved: {OUTPUT_DIR / 'throughput_vs_noise.pdf'}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating Manuscript Figures")
    print("=" * 60)

    # Load data
    print("\nLoading experimental results...")
    results = load_all_results()

    print(f"Found data for noise levels: {sorted(results.keys(), key=float)}")

    # Generate figures
    print("\nGenerating figures...")

    print("\n1. F1 Trajectories (Figure 1)...")
    figure1_f1_trajectories(results)

    print("\n2. Robustness Horizon (Figure 2)...")
    figure2_robustness_horizon(results)

    print("\n3. Failure Modes at ρ=0.55 (Figure 3)...")
    figure3_failure_modes(results)

    print("\n4. Throughput vs Noise (Figure 4)...")
    figure4_throughput_vs_noise(results)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
