@echo off
title PNAC - Synthetic Analysis
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================================
echo        PNAC - Synthetic Dataset Analysis
echo ============================================================================
echo.
echo This script runs a controlled synthetic analysis to validate PNAC:
echo.
echo   Experiment A: Noise Sweep (rho = 0.0 to 0.55)
echo   Experiment B: Difficulty Sweep (sigma = 0.3 to 1.1, rho = 0)
echo   Experiment C: Visual Mechanism (decision boundaries)
echo.
echo Output figures:
echo   - synthetic_noise_sweep.pdf
echo   - synthetic_difficulty_sweep.pdf
echo   - synthetic_mechanism.pdf
echo   - synthetic_summary.pdf
echo.
echo Expected runtime: ~2-3 minutes
echo.

call conda activate pnac
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'pnac'.
    pause
    exit /b 1
)

cd /d "%~dp0code"
python synthetic_pnac_analysis.py

echo.
echo ============================================================================
echo Done! Check the figures\ directory.
echo ============================================================================
pause
