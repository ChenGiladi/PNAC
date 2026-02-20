@echo off
rem ============================================================================
rem PNAC Publication Experiments with GPU (via WSL)
rem ============================================================================
title PNAC GPU Experiments
chcp 65001 >nul

echo ============================================================================
echo        PNAC PUBLICATION EXPERIMENTS (GPU via WSL)
echo ============================================================================
echo.
echo Starting WSL with GPU support...
echo.

cd /d "%~dp0"
wsl bash -c "cd \"$(wslpath '%cd%')\" && bash run_pnac_gpu.sh"

echo.
pause
