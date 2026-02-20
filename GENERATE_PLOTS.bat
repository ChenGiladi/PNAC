@echo off
title PNAC - Generate Plots (Conda)
chcp 65001 >nul

echo ============================================================================
echo        GENERATE PNAC PLOTS FROM RESULTS
echo ============================================================================
echo.

call conda activate pnac
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'pnac'.
    echo Please run SETUP_CONDA.bat first.
    pause
    exit /b 1
)

cd /d "%~dp0code"

echo Generating plots from experimental results...
python plot_pnac_results.py

echo.
echo Generating synthetic visualizations...
python generate_synthetic_plots.py

echo.
echo ============================================================================
echo Done! Check the figures/ directory for output.
echo ============================================================================
pause
