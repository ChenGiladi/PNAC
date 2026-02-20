@echo off
title PNAC - Analyze Studies
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================================
echo        PNAC STUDY ANALYSIS
echo ============================================================================
echo.

call conda activate pnac
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'pnac'.
    pause
    exit /b 1
)

cd /d "%~dp0code"
python analyze_studies.py

echo.
pause
