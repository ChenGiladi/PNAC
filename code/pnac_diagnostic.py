#!/usr/bin/env python3
"""
PNAC Diagnostic Script
----------------------
Implements the Pseudolabel Amplification Cascade protocol.
1. Injects synthetic noise into the training set.
2. Trains a baseline ResNet-18.
3. Iteratively pseudo-labels the unlabeled pool.
4. Tracks F1 decay on a CLEAN validation set.

This is an alias for pnac.py, provided for compatibility with the
Student Guide instructions. All functionality is identical.
"""

# Import and re-export everything from pnac.py
from pnac import *
from pnac import main

if __name__ == "__main__":
    main()
