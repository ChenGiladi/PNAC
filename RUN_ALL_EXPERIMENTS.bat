@echo off
REM ============================================================================
REM Run ALL PNAC experiments (all noise levels) using Conda
REM Skips already completed experiments automatically
REM ============================================================================

title PNAC - All Experiments (Conda)
chcp 65001 >nul

echo ============================================================================
echo        PNAC - RUNNING ALL NOISE LEVELS
echo ============================================================================
echo.
echo Configuration:
echo   Noise levels: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
echo   Seeds per level: 42, 123, 456
echo   Confidence threshold: 0.85
echo   Completed experiments will be skipped automatically
echo.
echo ============================================================================
echo.

cd /d "%~dp0"

call run_noise_conda.bat 0.0
call run_noise_conda.bat 0.1
call run_noise_conda.bat 0.2
call run_noise_conda.bat 0.3
call run_noise_conda.bat 0.4
call run_noise_conda.bat 0.5

echo.
echo ============================================================================
echo        ALL EXPERIMENTS COMPLETE
echo ============================================================================
echo.
echo Results saved to: code\pnac_results\
echo Run GENERATE_PLOTS.bat to create figures.
echo.
pause
