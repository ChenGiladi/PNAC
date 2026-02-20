@echo off
REM ============================================================================
REM TRANSITION STUDY: Map the F1 transition zone (0.45, 0.55, 0.60)
REM
REM Purpose: Characterize where exactly the F1 degradation occurs
REM          Current data shows plateau at 0.0-0.4, drop at 0.5
REM          This fills in the transition region
REM
REM Runs: 3 noise levels × 3 seeds = 9 experiments
REM ============================================================================

title PNAC - Transition Study
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================================
echo        PNAC TRANSITION STUDY
echo ============================================================================
echo.
echo This study maps the F1 transition zone with noise levels:
echo   0.45, 0.55, 0.60
echo.
echo Expected runtime: ~30-45 minutes (3 noise levels × 3 seeds)
echo.
echo Press any key to start or Ctrl+C to cancel...
pause >nul

echo.
echo [1/3] Running noise level 0.45...
call run_noise_conda.bat 0.45

echo.
echo [2/3] Running noise level 0.55...
call run_noise_conda.bat 0.55

echo.
echo [3/3] Running noise level 0.60...
call run_noise_conda.bat 0.60

echo.
echo ============================================================================
echo TRANSITION STUDY COMPLETE!
echo ============================================================================
echo.
echo Results saved to: code\pnac_results\noise_0.45, noise_0.55, noise_0.60
echo.
echo Next steps:
echo   1. Run the analysis to see the full transition curve
echo   2. If transition is sharp, consider adding 0.48, 0.52
echo.
pause
