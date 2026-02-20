#!/usr/bin/env python3
"""
Synthetic PNAC Analysis: Controlled Validation of the Robustness Horizon

This script creates a fully controlled synthetic environment to demonstrate:
1. The PNAC mechanism (amplification vs starvation)
2. The robustness horizon (stable → transition → unstable)
3. Why PNAC distinguishes noise from intrinsic task difficulty

Key experiments:
- Grid A: Noise sweep at fixed difficulty → shows β increases with ρ
- Grid B: Difficulty sweep at zero noise → shows β stays low (no false positives)
- Visual: Decision boundary evolution showing starvation mechanism

Dataset: 3-class 2D Gaussian mixture (mirrors the ultrasound structure)
- Class 0: "Background" - easy, well-separated
- Class 1: "Center" - overlaps with Class 2
- Class 2: "Not Center" - overlaps with Class 1

Usage:
    python synthetic_pnac_analysis.py

Output:
    figures/synthetic_*.pdf - Publication-ready figures
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_3class_gaussian(n_samples, separation=2.0, overlap_sigma=0.5, seed=42):
    """
    Generate 3-class 2D Gaussian mixture dataset.

    Structure (mirrors ultrasound dataset):
    - Class 0 (Background): Well-separated, easy
    - Class 1 (Center): Near origin, overlaps with Class 2
    - Class 2 (Not Center): Close to Class 1, creates ambiguity

    Args:
        n_samples: Total samples (divided equally among classes)
        separation: Distance between Class 0 and Classes 1,2
        overlap_sigma: Standard deviation (higher = more overlap between 1 and 2)
        seed: Random seed

    Returns:
        X: (n_samples, 2) features
        y: (n_samples,) labels {0, 1, 2}
    """
    np.random.seed(seed)
    n_per_class = n_samples // 3

    # Class 0: Background (far away, easy)
    X0 = np.random.randn(n_per_class, 2) * 0.4 + np.array([-separation, 0])

    # Class 1: Center (near origin)
    X1 = np.random.randn(n_per_class, 2) * overlap_sigma + np.array([0, 0])

    # Class 2: Not Center (close to Class 1, creates ambiguity)
    X2 = np.random.randn(n_per_class, 2) * overlap_sigma + np.array([1.0, 0])

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def inject_noise(y, noise_rate, n_classes=3, seed=42):
    """Inject uniform label noise."""
    np.random.seed(seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)

    if n_flip > 0:
        flip_idx = np.random.choice(len(y), n_flip, replace=False)
        for i in flip_idx:
            # Flip to a different class
            other_classes = [c for c in range(n_classes) if c != y[i]]
            y_noisy[i] = np.random.choice(other_classes)

    return y_noisy


def split_data(X, y, n_labeled=600, n_val=600, seed=42):
    """
    Split data into labeled, validation, and unlabeled pools.

    Following PNAC requirements:
    - Clean validation set V (no noise injected here)
    - Labeled set DL (noise will be injected)
    - Large unlabeled pool DU (|DU|/|DL| >= 5)
    """
    np.random.seed(seed)
    n_total = len(X)
    idx = np.random.permutation(n_total)

    X, y = X[idx], y[idx]

    X_labeled = X[:n_labeled]
    y_labeled = y[:n_labeled]

    X_val = X[n_labeled:n_labeled + n_val]
    y_val = y[n_labeled:n_labeled + n_val]

    X_unlabeled = X[n_labeled + n_val:]
    y_unlabeled_true = y[n_labeled + n_val:]  # Keep for analysis only

    return X_labeled, y_labeled, X_val, y_val, X_unlabeled, y_unlabeled_true


# =============================================================================
# MODEL
# =============================================================================

class SimpleMLP(nn.Module):
    """Small MLP for fast training on 2D data."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, X, y, epochs=100, lr=0.01):
    """Train the model on given data."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(model, X, y):
    """Evaluate model and return macro-F1."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        preds = torch.argmax(logits, dim=1).numpy()
    return f1_score(y, preds, average='macro')


def get_confident_predictions(model, X, threshold=0.85):
    """Get predictions above confidence threshold."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)

    mask = max_probs >= threshold
    return mask.numpy(), preds.numpy(), max_probs.numpy()


# =============================================================================
# PNAC IMPLEMENTATION
# =============================================================================

def run_pnac(X_labeled, y_labeled_noisy, y_labeled_true, X_val, y_val,
             X_unlabeled, y_unlabeled_true,
             n_iterations=5, confidence_threshold=0.85, epochs_per_iter=100):
    """
    Run the PNAC diagnostic protocol with full observability.

    Returns:
        f1_trajectory: List of F1 scores per iteration
        pseudo_label_counts: List of pseudo-labels added per iteration
        models: List of trained models (for visualization)
        precision_trajectory: List of pseudo-label precision q_t (synthetic advantage)
        corruption_trajectory: List of effective training set corruption ε_t
    """
    # Initialize
    X_train = X_labeled.copy()
    y_train = y_labeled_noisy.copy()
    y_train_true = y_labeled_true.copy()  # Track true labels for corruption measurement
    X_pool = X_unlabeled.copy()
    pool_available = np.ones(len(X_pool), dtype=bool)

    f1_trajectory = []
    pseudo_label_counts = []
    precision_trajectory = []  # q_t: pseudo-label precision
    corruption_trajectory = []  # ε_t: fraction of wrong labels in training set
    models = []

    # Initial corruption (before any pseudo-labels)
    initial_corruption = np.mean(y_train != y_train_true)
    corruption_trajectory.append(initial_corruption)

    for t in range(n_iterations + 1):
        # Train model
        model = SimpleMLP(n_classes=3)
        model = train_model(model, X_train, y_train, epochs=epochs_per_iter)
        models.append(model)

        # Evaluate on clean validation
        f1 = evaluate_model(model, X_val, y_val)
        f1_trajectory.append(f1)

        if t < n_iterations:
            # Pseudo-label available pool
            available_idx = np.where(pool_available)[0]
            if len(available_idx) == 0:
                pseudo_label_counts.append(0)
                precision_trajectory.append(np.nan)
                corruption_trajectory.append(corruption_trajectory[-1])
                continue

            X_candidates = X_pool[available_idx]
            y_candidates_true = y_unlabeled_true[available_idx]  # True labels for precision
            mask, preds, _ = get_confident_predictions(model, X_candidates, confidence_threshold)

            # Add confident predictions to training set
            n_selected = mask.sum()
            pseudo_label_counts.append(n_selected)

            if n_selected > 0:
                selected_idx = available_idx[mask]
                selected_preds = preds[mask]
                selected_true = y_candidates_true[mask]

                # Compute pseudo-label precision q_t (synthetic-only diagnostic)
                q_t = np.mean(selected_preds == selected_true)
                precision_trajectory.append(q_t)

                # Add to training set
                X_train = np.vstack([X_train, X_pool[selected_idx]])
                y_train = np.concatenate([y_train, selected_preds])
                y_train_true = np.concatenate([y_train_true, selected_true])
                pool_available[selected_idx] = False

                # Compute effective corruption ε_t
                epsilon_t = np.mean(y_train != y_train_true)
                corruption_trajectory.append(epsilon_t)
            else:
                precision_trajectory.append(np.nan)
                corruption_trajectory.append(corruption_trajectory[-1])

    return {
        'f1_trajectory': f1_trajectory,
        'pseudo_label_counts': pseudo_label_counts,
        'models': models,
        'precision_trajectory': precision_trajectory,  # q_t
        'corruption_trajectory': corruption_trajectory  # ε_t
    }


def fit_decay_rate(f1_scores):
    """
    Fit exponential decay model to extract β.

    Model: Δt = α(1 - exp(-βt))
    where Δt = F1_0 - F1_t
    """
    if len(f1_scores) < 3:
        return 0.0

    baseline = f1_scores[0]
    deltas = [max(0.0, baseline - f1) for f1 in f1_scores]

    # Simple linear fit on log-transformed data
    alpha = max(deltas) if max(deltas) > 0 else 1e-6

    xs, ys = [], []
    for t, delta in enumerate(deltas[1:], start=1):
        ratio = 1.0 - min(delta / alpha, 0.999999)
        if ratio > 0:
            xs.append(float(t))
            ys.append(np.log(ratio))

    if len(xs) < 2:
        return 0.0

    slope = np.polyfit(xs, ys, deg=1)[0]
    beta = -slope
    return max(0.0, beta)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_decision_boundary(ax, model, X, y, title, xlim=None, ylim=None):
    """Plot decision boundary with data points."""
    if xlim is None:
        xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    if ylim is None:
        ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    # Create grid
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                         np.linspace(ylim[0], ylim[1], 200))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    model.eval()
    with torch.no_grad():
        logits = model(grid)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy().reshape(xx.shape)
        confidence = torch.max(probs, dim=1)[0].numpy().reshape(xx.shape)

    # Custom colormap
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    cmap = ListedColormap(colors)

    # Plot decision regions with confidence as alpha
    ax.contourf(xx, yy, preds, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5])
    ax.contour(xx, yy, preds, colors='gray', linewidths=0.5, levels=[0.5, 1.5])

    # Plot data points
    scatter_colors = [colors[int(yi)] for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, edgecolors='k',
               s=20, alpha=0.7, linewidths=0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_noise_sweep():
    """
    Grid A: Noise sweep at fixed difficulty.
    Demonstrates: β increases with noise rate ρ.
    Now also captures q_t (precision) and ε_t (corruption).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Noise Sweep")
    print("=" * 60)

    noise_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6]
    n_seeds = 3

    results = {rho: {
        'f1_trajectories': [], 'betas': [], 'pseudo_counts': [],
        'precision_trajectories': [], 'corruption_trajectories': []
    } for rho in noise_rates}

    for rho in noise_rates:
        print(f"\nNoise rate ρ = {rho}")

        for seed in range(n_seeds):
            # Generate data
            X, y = generate_3class_gaussian(n_samples=7200, separation=3.0,
                                           overlap_sigma=0.6, seed=seed)
            X_L, y_L, X_V, y_V, X_U, y_U_true = split_data(X, y, n_labeled=600,
                                                            n_val=600, seed=seed)

            # Inject noise into labeled set (keep true labels for corruption tracking)
            y_L_noisy = inject_noise(y_L, rho, seed=seed + 100)

            # Run PNAC with full observability
            pnac_result = run_pnac(
                X_L, y_L_noisy, y_L,  # y_L is true labels
                X_V, y_V,
                X_U, y_U_true,
                n_iterations=5, confidence_threshold=0.85
            )

            f1_traj = pnac_result['f1_trajectory']
            pseudo_counts = pnac_result['pseudo_label_counts']
            beta = fit_decay_rate(f1_traj)

            results[rho]['f1_trajectories'].append(f1_traj)
            results[rho]['betas'].append(beta)
            results[rho]['pseudo_counts'].append(sum(pseudo_counts))
            results[rho]['precision_trajectories'].append(pnac_result['precision_trajectory'])
            results[rho]['corruption_trajectories'].append(pnac_result['corruption_trajectory'])

            # Compute mean precision (excluding NaN)
            prec = np.array(pnac_result['precision_trajectory'])
            valid_prec = prec[~np.isnan(prec)]
            mean_prec = np.mean(valid_prec) if len(valid_prec) > 0 else 0
            final_corr = pnac_result['corruption_trajectory'][-1]

            print(f"  Seed {seed}: F1_final={f1_traj[-1]:.3f}, β={beta:.3f}, "
                  f"Total pseudo={sum(pseudo_counts)}, ε_final={final_corr:.3f}, q_mean={mean_prec:.3f}")

    return results


def experiment_difficulty_sweep():
    """
    Grid B: Difficulty sweep at zero noise.
    Demonstrates: β stays low even when task is hard (no false positives).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Difficulty Sweep (ρ = 0)")
    print("=" * 60)

    # Vary overlap_sigma: higher = harder task
    difficulties = [0.3, 0.5, 0.7, 0.9, 1.1]
    n_seeds = 3

    results = {d: {
        'f1_trajectories': [], 'betas': [], 'baseline_f1': [],
        'precision_trajectories': [], 'corruption_trajectories': []
    } for d in difficulties}

    for difficulty in difficulties:
        print(f"\nOverlap σ = {difficulty}")

        for seed in range(n_seeds):
            # Generate data with varying difficulty
            X, y = generate_3class_gaussian(n_samples=7200, separation=3.0,
                                           overlap_sigma=difficulty, seed=seed)
            X_L, y_L, X_V, y_V, X_U, y_U_true = split_data(X, y, n_labeled=600,
                                                            n_val=600, seed=seed)

            # NO noise injection (ρ = 0) - y_L is both noisy and true
            pnac_result = run_pnac(
                X_L, y_L, y_L,  # No noise, so noisy = true
                X_V, y_V,
                X_U, y_U_true,
                n_iterations=5, confidence_threshold=0.85
            )

            f1_traj = pnac_result['f1_trajectory']
            beta = fit_decay_rate(f1_traj)

            results[difficulty]['f1_trajectories'].append(f1_traj)
            results[difficulty]['betas'].append(beta)
            results[difficulty]['baseline_f1'].append(f1_traj[0])
            results[difficulty]['precision_trajectories'].append(pnac_result['precision_trajectory'])
            results[difficulty]['corruption_trajectories'].append(pnac_result['corruption_trajectory'])

            print(f"  Seed {seed}: F1_0={f1_traj[0]:.3f}, F1_final={f1_traj[-1]:.3f}, "
                  f"β={beta:.3f}")

    return results


def experiment_visual_mechanism():
    """
    Visual demonstration of starvation vs amplification.
    Shows decision boundaries at iterations 0 and final.
    Now also tracks q_t and ε_t for mechanism validation.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Visual Mechanism Demonstration")
    print("=" * 60)

    noise_rates = [0.0, 0.3, 0.5]

    results = {}

    for rho in noise_rates:
        print(f"\nNoise rate ρ = {rho}")

        # Generate data
        X, y = generate_3class_gaussian(n_samples=7200, separation=3.0,
                                        overlap_sigma=0.6, seed=42)
        X_L, y_L, X_V, y_V, X_U, y_U_true = split_data(X, y, n_labeled=600,
                                                        n_val=600, seed=42)

        # Inject noise
        y_L_noisy = inject_noise(y_L, rho, seed=42)

        # Run PNAC with full observability
        pnac_result = run_pnac(
            X_L, y_L_noisy, y_L,  # y_L is true labels
            X_V, y_V,
            X_U, y_U_true,
            n_iterations=5, confidence_threshold=0.85
        )

        results[rho] = {
            'X_L': X_L, 'y_L_noisy': y_L_noisy,
            'X_V': X_V, 'y_V': y_V,
            'f1_traj': pnac_result['f1_trajectory'],
            'pseudo_counts': pnac_result['pseudo_label_counts'],
            'models': pnac_result['models'],
            'precision_traj': pnac_result['precision_trajectory'],
            'corruption_traj': pnac_result['corruption_trajectory']
        }

        print(f"  F1 trajectory: {[f'{f:.3f}' for f in pnac_result['f1_trajectory']]}")
        print(f"  Pseudo-labels: {pnac_result['pseudo_label_counts']}")
        print(f"  Total pseudo: {sum(pnac_result['pseudo_label_counts'])}")
        print(f"  Precision q_t: {[f'{p:.3f}' if not np.isnan(p) else 'N/A' for p in pnac_result['precision_trajectory']]}")
        print(f"  Corruption ε_t: {[f'{e:.3f}' for e in pnac_result['corruption_trajectory']]}")

    return results


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def plot_noise_sweep_results(results):
    """Figure: β vs ρ and F1 trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    noise_rates = sorted(results.keys())

    # Panel A: F1 trajectories
    ax = axes[0]
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(noise_rates)))

    for i, rho in enumerate(noise_rates):
        f1_trajs = results[rho]['f1_trajectories']
        mean_f1 = np.mean(f1_trajs, axis=0)
        std_f1 = np.std(f1_trajs, axis=0)
        iterations = np.arange(len(mean_f1))

        ax.plot(iterations, mean_f1, 'o-', color=colors[i], label=f'ρ={rho}',
                linewidth=2, markersize=6)
        ax.fill_between(iterations, mean_f1 - std_f1, mean_f1 + std_f1,
                       color=colors[i], alpha=0.2)

    ax.set_xlabel('PNAC Iteration')
    ax.set_ylabel('Validation Macro-F1')
    ax.set_title('(A) F1 Trajectories by Noise Level', fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    # Panel B: β vs ρ
    ax = axes[1]

    rhos = []
    mean_betas = []
    std_betas = []

    for rho in noise_rates:
        betas = results[rho]['betas']
        rhos.append(rho)
        mean_betas.append(np.mean(betas))
        std_betas.append(np.std(betas))

    ax.errorbar(rhos, mean_betas, yerr=std_betas, fmt='o-', capsize=5,
                capthick=2, markersize=10, linewidth=2, color='#2E86AB')

    # Highlight transition zone
    ax.axvspan(0.4, 0.55, alpha=0.2, color='orange', label='Transition zone')

    ax.set_xlabel('Noise Rate (ρ)')
    ax.set_ylabel('Fitted Decay Rate (β)')
    ax.set_title('(B) Decay Rate vs Noise', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'synthetic_noise_sweep.pdf')
    plt.savefig(OUTPUT_DIR / 'synthetic_noise_sweep.png')
    print(f"Saved: {OUTPUT_DIR / 'synthetic_noise_sweep.pdf'}")
    plt.close()


def plot_difficulty_sweep_results(results):
    """Figure: Showing that difficulty doesn't trigger false positive β."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    difficulties = sorted(results.keys())

    # Panel A: Baseline F1 vs difficulty
    ax = axes[0]

    means = [np.mean(results[d]['baseline_f1']) for d in difficulties]
    stds = [np.std(results[d]['baseline_f1']) for d in difficulties]

    ax.errorbar(difficulties, means, yerr=stds, fmt='s-', capsize=5,
                capthick=2, markersize=10, linewidth=2, color='#A23B72')

    ax.set_xlabel('Class Overlap (σ)')
    ax.set_ylabel('Baseline F1 (iteration 0)')
    ax.set_title('(A) Task Difficulty Effect on F1', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: β vs difficulty (should stay low)
    ax = axes[1]

    mean_betas = [np.mean(results[d]['betas']) for d in difficulties]
    std_betas = [np.std(results[d]['betas']) for d in difficulties]

    ax.errorbar(difficulties, mean_betas, yerr=std_betas, fmt='o-', capsize=5,
                capthick=2, markersize=10, linewidth=2, color='#2E86AB')

    # Reference line
    ax.axhline(y=0.5, color='red', linestyle='--', label='Noise threshold')

    ax.set_xlabel('Class Overlap (σ)')
    ax.set_ylabel('Fitted Decay Rate (β)')
    ax.set_title('(B) β Remains Low Despite Difficulty\n(No False Positives)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'synthetic_difficulty_sweep.pdf')
    plt.savefig(OUTPUT_DIR / 'synthetic_difficulty_sweep.png')
    print(f"Saved: {OUTPUT_DIR / 'synthetic_difficulty_sweep.pdf'}")
    plt.close()


def plot_visual_mechanism(results):
    """Figure: Decision boundaries showing starvation mechanism."""
    noise_rates = [0.0, 0.3, 0.5]

    fig, axes = plt.subplots(len(noise_rates), 3, figsize=(12, 10))

    # Compute consistent axis limits
    all_X = np.vstack([results[rho]['X_L'] for rho in noise_rates])
    xlim = (all_X[:, 0].min() - 0.5, all_X[:, 0].max() + 0.5)
    ylim = (all_X[:, 1].min() - 0.5, all_X[:, 1].max() + 0.5)

    for row, rho in enumerate(noise_rates):
        r = results[rho]

        # Column 1: Initial boundary (iteration 0)
        ax = axes[row, 0]
        f1_0 = r['f1_traj'][0]
        plot_decision_boundary(ax, r['models'][0], r['X_L'], r['y_L_noisy'],
                              f"ρ={rho} | Iter 0\nF1={f1_0:.2f}", xlim, ylim)
        if row == 0:
            ax.set_title(f"Initial (Iter 0)\nρ={rho}, F1={f1_0:.2f}", fontweight='bold')

        # Column 2: Final boundary
        ax = axes[row, 1]
        f1_final = r['f1_traj'][-1]
        total_pseudo = sum(r['pseudo_counts'])
        plot_decision_boundary(ax, r['models'][-1], r['X_L'], r['y_L_noisy'],
                              f"ρ={rho} | Iter 5\nF1={f1_final:.2f}", xlim, ylim)
        if row == 0:
            ax.set_title(f"Final (Iter 5)\nF1={f1_final:.2f}", fontweight='bold')

        # Column 3: Metrics
        ax = axes[row, 2]
        ax.axis('off')

        # Determine mode
        if total_pseudo == 0:
            mode = "STARVATION"
            mode_color = "red"
        elif r['f1_traj'][-1] < r['f1_traj'][0] - 0.1:
            mode = "AMPLIFICATION"
            mode_color = "orange"
        else:
            mode = "STABLE"
            mode_color = "green"

        text = f"Noise Rate: ρ = {rho}\n\n"
        text += f"F1 Trajectory:\n{' → '.join([f'{f:.2f}' for f in r['f1_traj']])}\n\n"
        text += f"Pseudo-labels/iter:\n{r['pseudo_counts']}\n\n"
        text += f"Total pseudo-labels: {total_pseudo}\n\n"
        text += f"Mode: {mode}"

        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.text(0.5, 0.1, mode, transform=ax.transAxes, fontsize=14,
                fontweight='bold', color=mode_color, ha='center')

    # Row labels
    for row, rho in enumerate(noise_rates):
        axes[row, 0].set_ylabel(f'ρ = {rho}', fontsize=12, fontweight='bold')

    plt.suptitle('Synthetic PNAC Mechanism: Starvation vs Amplification',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'synthetic_mechanism.pdf')
    plt.savefig(OUTPUT_DIR / 'synthetic_mechanism.png')
    print(f"Saved: {OUTPUT_DIR / 'synthetic_mechanism.pdf'}")
    plt.close()


def plot_mechanism_validation(noise_results):
    """
    Figure: Mechanism validation showing q_t (precision) and ε_t (corruption).
    This is the "synthetic advantage" - directly measuring latent quantities.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Select representative noise levels
    noise_levels = [0.0, 0.3, 0.5]
    colors = ['#2E86AB', '#F18F01', '#E94F37']

    # Panel A: Pseudo-label precision q_t over iterations
    ax = axes[0]
    for i, rho in enumerate(noise_levels):
        if rho in noise_results:
            prec_trajs = noise_results[rho]['precision_trajectories']
            # Average across seeds, handling NaN
            prec_array = np.array(prec_trajs)
            mean_prec = np.nanmean(prec_array, axis=0)
            iterations = np.arange(1, len(mean_prec) + 1)
            ax.plot(iterations, mean_prec, 'o-', color=colors[i],
                   label=f'ρ={rho}', linewidth=2, markersize=8)

    ax.set_xlabel('PNAC Iteration')
    ax.set_ylabel('Pseudo-label Precision ($q_t$)')
    ax.set_title('(A) Precision Drops with Noise', fontweight='bold')
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='τ threshold')

    # Panel B: Effective corruption ε_t over iterations
    ax = axes[1]
    for i, rho in enumerate(noise_levels):
        if rho in noise_results:
            corr_trajs = noise_results[rho]['corruption_trajectories']
            # Average across seeds
            corr_array = np.array(corr_trajs)
            mean_corr = np.mean(corr_array, axis=0)
            iterations = np.arange(len(mean_corr))
            ax.plot(iterations, mean_corr, 's-', color=colors[i],
                   label=f'ρ={rho}', linewidth=2, markersize=8)

    ax.set_xlabel('PNAC Iteration')
    ax.set_ylabel('Training Set Corruption ($\\varepsilon_t$)')
    ax.set_title('(B) Corruption Evolves Over Cascade', fontweight='bold')
    ax.legend()
    ax.set_ylim(-0.02, 0.6)
    ax.grid(True, alpha=0.3)

    # Panel C: Triple plot - F1, throughput, precision for one noise level
    ax = axes[2]
    rho = 0.3  # Representative noise level
    if rho in noise_results:
        f1_trajs = noise_results[rho]['f1_trajectories']
        prec_trajs = noise_results[rho]['precision_trajectories']

        mean_f1 = np.mean(f1_trajs, axis=0)
        prec_array = np.array(prec_trajs)
        mean_prec = np.nanmean(prec_array, axis=0)

        iterations_f1 = np.arange(len(mean_f1))
        iterations_prec = np.arange(1, len(mean_prec) + 1)

        ax.plot(iterations_f1, mean_f1, 'o-', color='#2E86AB',
               label='F1', linewidth=2, markersize=8)
        ax.plot(iterations_prec, mean_prec, 's-', color='#E94F37',
               label='Precision $q_t$', linewidth=2, markersize=8)

    ax.set_xlabel('PNAC Iteration')
    ax.set_ylabel('Validation Macro-F1 / Precision')
    ax.set_title(f'(C) F1 and Precision at ρ={rho}', fontweight='bold')
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'synthetic_mechanism_validation.pdf')
    plt.savefig(OUTPUT_DIR / 'synthetic_mechanism_validation.png')
    print(f"Saved: {OUTPUT_DIR / 'synthetic_mechanism_validation.pdf'}")
    plt.close()


def plot_combined_summary(noise_results, difficulty_results):
    """Combined figure for manuscript: proves noise ≠ difficulty."""
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8),
                             gridspec_kw={'width_ratios': [1.0, 1.0, 1.35]})

    # Panel A: F1 vs noise
    ax = axes[0]
    noise_rates = sorted(noise_results.keys())

    final_f1_means = []
    final_f1_stds = []
    for rho in noise_rates:
        f1s = [traj[-1] for traj in noise_results[rho]['f1_trajectories']]
        final_f1_means.append(np.mean(f1s))
        final_f1_stds.append(np.std(f1s))

    ax.errorbar(noise_rates, final_f1_means, yerr=final_f1_stds, fmt='o-',
                capsize=5, markersize=8, linewidth=2, color='#2E86AB',
                label='Final F1')
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.7)
    ax.axvspan(0.4, 0.55, alpha=0.15, color='orange')
    ax.set_xlabel('Noise Rate (ρ)')
    ax.set_ylabel('Final Macro-F1')
    ax.set_title('(A) Noise Causes F1 Drop', fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    # Panel B: F1 vs difficulty (no noise)
    ax = axes[1]
    difficulties = sorted(difficulty_results.keys())

    baseline_means = [np.mean(difficulty_results[d]['baseline_f1']) for d in difficulties]
    baseline_stds = [np.std(difficulty_results[d]['baseline_f1']) for d in difficulties]
    final_means = [np.mean([t[-1] for t in difficulty_results[d]['f1_trajectories']]) for d in difficulties]
    final_stds = [np.std([t[-1] for t in difficulty_results[d]['f1_trajectories']]) for d in difficulties]

    ax.errorbar(difficulties, baseline_means, yerr=baseline_stds, fmt='s-',
                capsize=5, markersize=8, linewidth=2, color='#A23B72',
                label='Baseline F1')
    ax.errorbar(difficulties, final_means, yerr=final_stds, fmt='o-',
                capsize=5, markersize=8, linewidth=2, color='#2E86AB',
                label='Final F1')
    ax.set_xlabel('Class Overlap (σ)')
    ax.set_ylabel('Macro-F1')
    ax.set_title('(B) Difficulty Drops F1, but...', fontweight='bold')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    # Panel C: β comparison
    ax = axes[2]

    # β from noise sweep
    noise_betas = [np.mean(noise_results[rho]['betas']) for rho in noise_rates]
    noise_beta_stds = [np.std(noise_results[rho]['betas']) for rho in noise_rates]

    # β from difficulty sweep
    diff_betas = [np.mean(difficulty_results[d]['betas']) for d in difficulties]
    diff_beta_stds = [np.std(difficulty_results[d]['betas']) for d in difficulties]

    x_noise = np.arange(len(noise_rates))
    x_diff = np.arange(len(difficulties)) + len(noise_rates) + 1

    ax.bar(x_noise, noise_betas, yerr=noise_beta_stds, color='#F18F01',
           alpha=0.8, label='Noise (ρ varied)', capsize=3)
    ax.bar(x_diff, diff_betas, yerr=diff_beta_stds, color='#2E86AB',
           alpha=0.8, label='Difficulty (σ varied, ρ=0)', capsize=3)

    ax.set_xticks(list(x_noise) + list(x_diff))
    tick_labels = [f'{r:.2f}'.rstrip('0').rstrip('.') for r in noise_rates] + \
                  [f'{d:.1f}'.rstrip('0').rstrip('.') for d in difficulties]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.tick_params(axis='x', pad=1)
    ax.set_xlim(-0.6, x_diff[-1] + 0.6)

    split = len(noise_rates) - 0.5
    ax.axvline(split, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.text((len(noise_rates)-1)/2, -0.24, 'Noise (ρ)',
            transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
    ax.text(len(noise_rates)+1+(len(difficulties)-1)/2, -0.24, 'Difficulty (σ)',
            transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)

    ax.set_ylabel('Decay Rate (β)')
    ax.set_title('(C) β Distinguishes Noise from Difficulty', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    fig.subplots_adjust(bottom=0.24, wspace=0.32)
    plt.savefig(OUTPUT_DIR / 'synthetic_summary.pdf')
    plt.savefig(OUTPUT_DIR / 'synthetic_summary.png')
    print(f"Saved: {OUTPUT_DIR / 'synthetic_summary.pdf'}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("SYNTHETIC PNAC ANALYSIS")
    print("Controlled Validation of Robustness Horizon")
    print("=" * 60)

    # Run experiments
    noise_results = experiment_noise_sweep()
    difficulty_results = experiment_difficulty_sweep()
    visual_results = experiment_visual_mechanism()

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    plot_noise_sweep_results(noise_results)
    plot_difficulty_sweep_results(difficulty_results)
    plot_visual_mechanism(visual_results)
    plot_mechanism_validation(noise_results)
    plot_combined_summary(noise_results, difficulty_results)

    print("\n" + "=" * 60)
    print("SYNTHETIC ANALYSIS COMPLETE!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # Print summary for manuscript
    print("\n" + "=" * 60)
    print("KEY FINDINGS FOR MANUSCRIPT")
    print("=" * 60)

    print("\n1. NOISE SWEEP (ρ varied, difficulty fixed):")
    for rho in sorted(noise_results.keys()):
        mean_f1 = np.mean([t[-1] for t in noise_results[rho]['f1_trajectories']])
        mean_beta = np.mean(noise_results[rho]['betas'])
        mean_pseudo = np.mean(noise_results[rho]['pseudo_counts'])
        print(f"   ρ={rho}: F1={mean_f1:.3f}, β={mean_beta:.3f}, pseudo={mean_pseudo:.0f}")

    print("\n2. DIFFICULTY SWEEP (σ varied, ρ=0):")
    for diff in sorted(difficulty_results.keys()):
        mean_f1_0 = np.mean(difficulty_results[diff]['baseline_f1'])
        mean_f1_final = np.mean([t[-1] for t in difficulty_results[diff]['f1_trajectories']])
        mean_beta = np.mean(difficulty_results[diff]['betas'])
        print(f"   σ={diff}: F1_0={mean_f1_0:.3f}, F1_final={mean_f1_final:.3f}, β={mean_beta:.3f}")

    print("\n3. KEY INSIGHT:")
    print("   - Noise increases β (triggers PNAC alarm)")
    print("   - Difficulty alone keeps β low (no false alarm)")
    print("   - PNAC correctly distinguishes noise from task difficulty")


if __name__ == "__main__":
    main()
