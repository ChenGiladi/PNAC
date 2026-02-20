@echo off
title PNAC - Generate Manuscript Figures
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================================
echo        PNAC - Generate Manuscript Figures
echo ============================================================================
echo.
echo This script generates publication-quality figures for the manuscript:
echo   1. f1_trajectories.pdf     - F1 across iterations by noise level
echo   2. robustness_horizon.pdf  - Mean F1 vs noise showing transition
echo   3. failure_modes.pdf       - Starvation vs Amplification at rho=0.55
echo   4. throughput_vs_noise.pdf - Pseudo-label counts vs noise
echo.
echo Output directory: figures\
echo.

call conda activate pnac
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'pnac'.
    pause
    exit /b 1
)

cd /d "%~dp0code"
python generate_manuscript_figures.py

echo.
echo ============================================================================
echo Done! Check the figures\ directory.
echo ============================================================================
pause
