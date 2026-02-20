#!/usr/bin/env python3
"""
Clear PNAC Results
------------------
Removes all experiment results to allow starting fresh.
"""

import shutil
import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Directories to clear
RESULTS_DIR = SCRIPT_DIR / "pnac_results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def main():
    print("=" * 60)
    print("CLEAR PNAC RESULTS")
    print("=" * 60)
    print()

    # Check what exists
    results_exist = RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir())
    figures_exist = FIGURES_DIR.exists() and any(FIGURES_DIR.iterdir())

    if not results_exist and not figures_exist:
        print("Nothing to clear - no results or figures found.")
        return

    # Show what will be deleted
    if results_exist:
        # Count completed experiments
        count = len(list(RESULTS_DIR.glob("**/summary.json")))
        print(f"Results directory: {RESULTS_DIR}")
        print(f"  Contains {count} completed experiments")

    if figures_exist:
        fig_count = len(list(FIGURES_DIR.glob("*.pdf")))
        print(f"Figures directory: {FIGURES_DIR}")
        print(f"  Contains {fig_count} PDF figures")

    print()

    # Confirm deletion
    response = input("Delete all results and figures? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return

    # Delete
    if results_exist:
        print(f"Deleting {RESULTS_DIR}...")
        shutil.rmtree(RESULTS_DIR)

    if figures_exist:
        print(f"Deleting {FIGURES_DIR}...")
        shutil.rmtree(FIGURES_DIR)

    print()
    print("Done! All results cleared.")
    print("Run 'python run_calibration.py' to start fresh experiments.")


if __name__ == "__main__":
    main()
