#!/usr/bin/env python3
"""
Analyze results from PNAC Transition and Confidence studies.

This script:
1. Loads results from both studies
2. Generates summary tables
3. Creates visualization plots
4. Identifies the transition point and confidence sensitivity

Usage:
    python analyze_studies.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "pnac_results"
CONFIDENCE_RESULTS_DIR = SCRIPT_DIR / "pnac_results_confidence_study"
OUTPUT_DIR = SCRIPT_DIR.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_results(results_dir: Path) -> dict:
    """Load all results from a results directory."""
    results = defaultdict(lambda: defaultdict(dict))

    for noise_dir in results_dir.glob("noise_*"):
        noise_level = noise_dir.name.replace("noise_", "")
        for seed_dir in noise_dir.glob("seed_*"):
            seed = seed_dir.name.replace("seed_", "")
            # Find the latest run with summary.json
            runs = sorted(seed_dir.iterdir(), key=lambda x: x.name, reverse=True)
            for run in runs:
                summary_file = run / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        data = json.load(f)
                        results[noise_level][seed] = {
                            'f1_scores': data['f1_scores'],
                            'final_f1': data['f1_scores'][-1],
                            'beta': data.get('decay_beta', None),
                            'confidence': data.get('confidence', 0.85)
                        }
                    break
    return dict(results)


def load_confidence_results(results_dir: Path) -> dict:
    """Load results from confidence study directory."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for conf_dir in results_dir.glob("conf_*"):
        confidence = conf_dir.name.replace("conf_", "")
        for noise_dir in conf_dir.glob("noise_*"):
            noise_level = noise_dir.name.replace("noise_", "")
            for seed_dir in noise_dir.glob("seed_*"):
                seed = seed_dir.name.replace("seed_", "")
                runs = sorted(seed_dir.iterdir(), key=lambda x: x.name, reverse=True)
                for run in runs:
                    summary_file = run / "summary.json"
                    if summary_file.exists():
                        with open(summary_file) as f:
                            data = json.load(f)
                            results[confidence][noise_level][seed] = {
                                'f1_scores': data['f1_scores'],
                                'final_f1': data['f1_scores'][-1],
                                'beta': data.get('decay_beta', None)
                            }
                        break
    return dict(results)


def analyze_transition(results: dict):
    """Analyze the transition study results."""
    print("\n" + "=" * 70)
    print("TRANSITION STUDY RESULTS")
    print("=" * 70)

    # Sort noise levels
    noise_levels = sorted(results.keys(), key=float)

    print(f"\n{'Noise':<10} {'seed_42':<12} {'seed_123':<12} {'seed_456':<12} {'Mean':<12} {'Std':<10}")
    print("-" * 70)

    summary_data = []
    for noise in noise_levels:
        seeds = results[noise]
        f1_values = [seeds[s]['final_f1'] for s in ['42', '123', '456'] if s in seeds]

        if f1_values:
            mean_f1 = np.mean(f1_values)
            std_f1 = np.std(f1_values)
            summary_data.append((float(noise), mean_f1, std_f1))

            s42 = f"{seeds.get('42', {}).get('final_f1', 0):.4f}" if '42' in seeds else "MISS"
            s123 = f"{seeds.get('123', {}).get('final_f1', 0):.4f}" if '123' in seeds else "MISS"
            s456 = f"{seeds.get('456', {}).get('final_f1', 0):.4f}" if '456' in seeds else "MISS"

            print(f"{noise:<10} {s42:<12} {s123:<12} {s456:<12} {mean_f1:<12.4f} {std_f1:<10.4f}")

    # Find transition point
    if len(summary_data) >= 2:
        print("\n" + "-" * 70)
        print("TRANSITION ANALYSIS:")

        # Find largest drop between consecutive noise levels
        max_drop = 0
        transition_point = None
        for i in range(1, len(summary_data)):
            drop = summary_data[i-1][1] - summary_data[i][1]
            if drop > max_drop:
                max_drop = drop
                transition_point = (summary_data[i-1][0], summary_data[i][0])

        if transition_point:
            print(f"  Largest F1 drop: {max_drop:.4f}")
            print(f"  Transition between: noise {transition_point[0]} → {transition_point[1]}")

    return summary_data


def analyze_confidence(results: dict):
    """Analyze the confidence study results."""
    print("\n" + "=" * 70)
    print("CONFIDENCE THRESHOLD STUDY RESULTS")
    print("=" * 70)

    confidence_levels = sorted(results.keys(), key=float)

    for conf in confidence_levels:
        print(f"\n--- Confidence Threshold: {conf} ---")
        noise_levels = sorted(results[conf].keys(), key=float)

        print(f"{'Noise':<10} {'Mean F1':<12} {'Std':<10}")
        print("-" * 35)

        for noise in noise_levels:
            seeds = results[conf][noise]
            f1_values = [seeds[s]['final_f1'] for s in ['42', '123', '456'] if s in seeds]

            if f1_values:
                mean_f1 = np.mean(f1_values)
                std_f1 = np.std(f1_values)
                print(f"{noise:<10} {mean_f1:<12.4f} {std_f1:<10.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("CONFIDENCE SENSITIVITY SUMMARY")
    print("=" * 70)
    print("\nF1 at noise=0.5 by confidence threshold:")

    for conf in confidence_levels:
        if '0.5' in results[conf]:
            seeds = results[conf]['0.5']
            f1_values = [seeds[s]['final_f1'] for s in ['42', '123', '456'] if s in seeds]
            if f1_values:
                print(f"  τ = {conf}: F1 = {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")


def plot_transition_curve(results: dict):
    """Plot the F1 vs noise curve with transition zone highlighted."""
    noise_levels = sorted(results.keys(), key=float)

    means = []
    stds = []
    noises = []

    for noise in noise_levels:
        seeds = results[noise]
        f1_values = [seeds[s]['final_f1'] for s in ['42', '123', '456'] if s in seeds]
        if f1_values:
            noises.append(float(noise))
            means.append(np.mean(f1_values))
            stds.append(np.std(f1_values))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(noises, means, yerr=stds, fmt='o-', capsize=5, capthick=2,
                markersize=8, linewidth=2, color='#2E86AB')

    # Highlight transition zone (0.4 - 0.6)
    ax.axvspan(0.4, 0.6, alpha=0.2, color='red', label='Transition Zone')

    # Add threshold line
    ax.axhline(y=0.90, color='orange', linestyle='--', label='Threshold (F1=0.90)')

    ax.set_xlabel('Noise Rate (ρ)', fontsize=12)
    ax.set_ylabel('Final Macro-F1', fontsize=12)
    ax.set_title('PNAC F1 Degradation vs Noise Rate', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(noises) + 0.05)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'transition_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'transition_curve.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'transition_curve.pdf'}")
    plt.close()


def plot_confidence_comparison(results: dict):
    """Plot F1 vs noise for different confidence thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    confidence_levels = sorted(results.keys(), key=float)

    for conf, color in zip(confidence_levels, colors):
        noise_levels = sorted(results[conf].keys(), key=float)

        noises = []
        means = []
        stds = []

        for noise in noise_levels:
            seeds = results[conf][noise]
            f1_values = [seeds[s]['final_f1'] for s in ['42', '123', '456'] if s in seeds]
            if f1_values:
                noises.append(float(noise))
                means.append(np.mean(f1_values))
                stds.append(np.std(f1_values))

        if noises:
            noises = np.array(noises)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(noises, means, 'o-', label=f'τ = {conf}', color=color,
                    markersize=7, linewidth=2)
            ax.fill_between(noises, means - stds, means + stds,
                           color=color, alpha=0.15)

    ax.set_xlabel('Noise Rate (ρ)', fontsize=12)
    ax.set_ylabel('Final Macro-F1', fontsize=12)
    ax.legend(title='Confidence Threshold (τ)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'confidence_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'confidence_comparison.pdf'}")
    plt.close()


def main():
    print("=" * 70)
    print("PNAC STUDY ANALYSIS")
    print("=" * 70)

    # Load and analyze transition study
    if RESULTS_DIR.exists():
        results = load_results(RESULTS_DIR)
        if results:
            summary = analyze_transition(results)
            plot_transition_curve(results)
        else:
            print(f"\nNo results found in {RESULTS_DIR}")
    else:
        print(f"\nResults directory not found: {RESULTS_DIR}")

    # Load and analyze confidence study
    if CONFIDENCE_RESULTS_DIR.exists():
        conf_results = load_confidence_results(CONFIDENCE_RESULTS_DIR)
        if conf_results:
            analyze_confidence(conf_results)
            plot_confidence_comparison(conf_results)
        else:
            print(f"\nNo results found in {CONFIDENCE_RESULTS_DIR}")
    else:
        print(f"\nConfidence study directory not found: {CONFIDENCE_RESULTS_DIR}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
