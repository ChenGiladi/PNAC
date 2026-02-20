#!/usr/bin/env python3
"""
Generate Geometric Intuition Visualization for PNAC Manuscript.

This script creates a 2x2 visualization showing decision boundary evolution
on a 2D toy dataset (Two Moons) to demonstrate the error amplification mechanism.

Rows: Iteration 0 (Initial) vs Iteration 5 (Final)
Columns: Clean Labels vs Noisy Labels (rho=0.3)

The visualization shows how the decision boundary distorts to "memorize"
mislabeled points in the noisy case, providing geometric evidence for
the error amplification dynamics derived in Section 5.

Output: figures/toy_mechanism.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SAMPLES = 300
NOISE_LEVEL = 0.2  # Intrinsic noise in make_moons
RHO = 0.3  # Label noise rate for noisy scenario
N_ITERATIONS = 5
CONFIDENCE_THRESHOLD = 0.7  # Lower threshold for toy example to see effect


def generate_data():
    """Generate Two Moons dataset with clean and noisy labels."""
    X, y_clean = make_moons(n_samples=N_SAMPLES, noise=NOISE_LEVEL, random_state=42)

    # Create noisy labels by flipping RHO fraction
    y_noisy = y_clean.copy()
    n_flip = int(RHO * len(y_clean))
    flip_indices = np.random.choice(len(y_clean), n_flip, replace=False)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

    return X, y_clean, y_noisy, flip_indices


def create_unlabeled_pool():
    """Create unlabeled pool for pseudo-labeling."""
    X_unlabeled, _ = make_moons(n_samples=500, noise=NOISE_LEVEL, random_state=123)
    return X_unlabeled


def train_model(X, y):
    """Train a simple classifier (SVC with RBF kernel)."""
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X, y)
    return model


def pseudo_label_iteration(model, X_train, y_train, X_unlabeled, threshold=CONFIDENCE_THRESHOLD):
    """
    Perform one iteration of pseudo-labeling.
    Returns updated training set and model.
    """
    # Get predictions and confidence for unlabeled data
    probs = model.predict_proba(X_unlabeled)
    max_probs = probs.max(axis=1)
    pseudo_labels = model.predict(X_unlabeled)

    # Select high-confidence samples
    confident_mask = max_probs >= threshold

    if confident_mask.sum() > 0:
        X_pseudo = X_unlabeled[confident_mask]
        y_pseudo = pseudo_labels[confident_mask]

        # Add to training set
        X_new = np.vstack([X_train, X_pseudo])
        y_new = np.concatenate([y_train, y_pseudo])
    else:
        X_new, y_new = X_train, y_train

    # Train new model
    new_model = train_model(X_new, y_new)

    return new_model, X_new, y_new


def run_pnac_simulation(X, y, X_unlabeled, n_iterations=N_ITERATIONS):
    """
    Run PNAC simulation and return models at iteration 0 and final iteration.
    """
    # Initial model
    model_0 = train_model(X, y)

    # Run iterations
    current_model = model_0
    X_train = X.copy()
    y_train = y.copy()

    for i in range(n_iterations):
        current_model, X_train, y_train = pseudo_label_iteration(
            current_model, X_train, y_train, X_unlabeled
        )

    model_final = current_model

    return model_0, model_final


def plot_decision_boundary(ax, model, X, y, title, flip_indices=None, show_mislabeled=False):
    """
    Plot decision boundary with scatter points.
    """
    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Get decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions with custom colormap
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

    # Plot decision boundary contour
    ax.contour(xx, yy, Z, levels=[0.5], colors='gray', linewidths=2, linestyles='--')

    # Plot points
    colors = ['#0000CC', '#CC0000']  # Dark blue, dark red
    markers = ['o', 's']

    for class_idx in [0, 1]:
        mask = y == class_idx
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx],
                   marker=markers[class_idx], s=30, edgecolor='white',
                   linewidth=0.5, label=f'Class {class_idx}')

    # Highlight mislabeled points if requested
    if show_mislabeled and flip_indices is not None:
        ax.scatter(X[flip_indices, 0], X[flip_indices, 1],
                   facecolors='none', edgecolors='lime',
                   s=100, linewidths=2, label='Mislabeled')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    """Generate the 2x2 visualization figure."""
    print("Generating Two Moons toy visualization for PNAC...")

    # Generate data
    X, y_clean, y_noisy, flip_indices = generate_data()
    X_unlabeled = create_unlabeled_pool()

    print(f"  Dataset: {N_SAMPLES} samples")
    print(f"  Label noise rate (rho): {RHO}")
    print(f"  Number of mislabeled samples: {len(flip_indices)}")
    print(f"  Unlabeled pool: {len(X_unlabeled)} samples")
    print(f"  Running {N_ITERATIONS} PNAC iterations...")

    # Run PNAC for clean scenario
    print("  Training clean scenario...")
    model_clean_0, model_clean_final = run_pnac_simulation(X, y_clean, X_unlabeled)

    # Run PNAC for noisy scenario
    print("  Training noisy scenario...")
    model_noisy_0, model_noisy_final = run_pnac_simulation(X, y_noisy, X_unlabeled)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    # Row 0: Iteration 0
    # Column 0: Clean Labels
    plot_decision_boundary(axes[0, 0], model_clean_0, X, y_clean,
                           'Clean Labels - Iteration 0')

    # Column 1: Noisy Labels
    plot_decision_boundary(axes[0, 1], model_noisy_0, X, y_noisy,
                           f'Noisy Labels ($\\rho$={RHO}) - Iteration 0',
                           flip_indices=flip_indices, show_mislabeled=True)

    # Row 1: Final Iteration
    # Column 0: Clean Labels
    plot_decision_boundary(axes[1, 0], model_clean_final, X, y_clean,
                           f'Clean Labels - Iteration {N_ITERATIONS}')

    # Column 1: Noisy Labels
    plot_decision_boundary(axes[1, 1], model_noisy_final, X, y_noisy,
                           f'Noisy Labels ($\\rho$={RHO}) - Iteration {N_ITERATIONS}',
                           flip_indices=flip_indices, show_mislabeled=True)

    # Add row labels
    axes[0, 0].set_ylabel('Initial', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('After 5 Iterations', fontsize=12, fontweight='bold')

    # Add legend
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))

    # Add annotations
    fig.text(0.27, 0.95, 'Clean Labels', ha='center', va='bottom',
             fontsize=12, fontweight='bold')
    fig.text(0.73, 0.95, 'Noisy Labels', ha='center', va='bottom',
             fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    # Save figure
    output_path = '../figures/toy_mechanism.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\n  Figure saved to: {output_path}")

    # Also save PNG for quick preview
    png_path = '../figures/toy_mechanism.png'
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
    print(f"  Preview saved to: {png_path}")

    plt.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
