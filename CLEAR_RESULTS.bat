@echo off
rem ============================================================================
rem Clear PNAC Results - Start Fresh
rem ============================================================================
title Clear PNAC Results
chcp 65001 >nul

echo ============================================================================
echo        CLEAR PNAC RESULTS
echo ============================================================================
echo.
echo This will delete all experiment results and figures.
echo.

cd /d "%~dp0\code"
python clear_results.py

echo.
pause
