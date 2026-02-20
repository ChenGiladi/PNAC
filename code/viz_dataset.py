#!/usr/bin/env python3
"""
Generate a publication-quality figure displaying representative ultrasound images
from the three-class dataset (background, center, not_center).

This figure is intended for Section 7.1 (Dataset) of the manuscript to provide
visual evidence of the dataset described in the text.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

# Configure matplotlib for publication-quality figures (same style as generate_synthetic_plots.py)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory (relative to the project structure)
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'code 04112025', 'three_classes_dataset')

# Class directories and display names
CLASSES = [
    ('background', 'Background'),
    ('center', 'Center'),
    ('not_center', 'Not Center'),
]


def get_representative_image(class_dir):
    """
    Select a representative image from a class directory.

    Strategy: Pick an image from the middle of the sorted list to avoid
    edge cases and get a typical example.

    Parameters:
        class_dir: Path to the class directory

    Returns:
        Path to the selected image
    """
    images = sorted([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not images:
        raise FileNotFoundError(f"No images found in {class_dir}")

    # Select from middle of the list for a representative sample
    mid_idx = len(images) // 2
    return os.path.join(class_dir, images[mid_idx])


def generate_dataset_samples_figure():
    """Generate a 1x3 grid of representative ultrasound images from each class."""

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, (class_name, display_name) in zip(axes, CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            ax.text(0.5, 0.5, f'{display_name}\n(not found)',
                    ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Get representative image
        img_path = get_representative_image(class_dir)
        img = Image.open(img_path)

        # Display image
        ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        ax.set_title(display_name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

    plt.tight_layout()

    # Save as PDF
    output_path = os.path.join(OUTPUT_DIR, 'dataset_samples.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate the dataset samples figure."""
    print("Generating dataset samples visualization...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Please ensure the three_classes_dataset directory is available.")
        return None

    # Generate the figure
    output_path = generate_dataset_samples_figure()

    print()
    print("Done!")
    print(f"Generated figure: {output_path}")
    return output_path


if __name__ == '__main__':
    main()
