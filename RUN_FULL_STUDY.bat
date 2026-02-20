@echo off
REM ============================================================================
REM FULL PNAC STUDY: Transition + Confidence Threshold Analysis
REM
REM Part 1: Transition Study (MINIMUM)
REM   - New noise levels: 0.45, 0.55, 0.60
REM   - Uses default confidence (0.85)
REM   - 9 experiments
REM
REM Part 2: Confidence Study (BETTER)
REM   - Noise levels: 0.0, 0.2, 0.4, 0.5, 0.6
REM   - Confidence: 0.70, 0.80, 0.90, 0.95
REM   - 60 experiments
REM
REM Total: 69 experiments
REM Expected runtime: ~6-7 hours
REM ============================================================================

title PNAC - Full Study
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================================
echo        PNAC FULL STUDY
echo ============================================================================
echo.
echo This runs BOTH studies:
echo.
echo   PART 1 - Transition Study (30-45 min)
echo     Noise: 0.45, 0.55, 0.60
echo     9 experiments
echo.
echo   PART 2 - Confidence Study (5-6 hours)
echo     Noise: 0.0, 0.2, 0.4, 0.5, 0.6
echo     Confidence: 0.70, 0.80, 0.90, 0.95
echo     60 experiments
echo.
echo Total: 69 experiments, ~6-7 hours
echo.
echo Press any key to start or Ctrl+C to cancel...
pause >nul

echo.
echo ############################################################################
echo PART 1: TRANSITION STUDY
echo ############################################################################
echo.

call run_noise_conda.bat 0.45
call run_noise_conda.bat 0.55
call run_noise_conda.bat 0.60

echo.
echo ############################################################################
echo PART 2: CONFIDENCE STUDY
echo ############################################################################
echo.

REM Confidence 0.70
call run_noise_confidence_conda.bat 0.0 0.70
call run_noise_confidence_conda.bat 0.2 0.70
call run_noise_confidence_conda.bat 0.4 0.70
call run_noise_confidence_conda.bat 0.5 0.70
call run_noise_confidence_conda.bat 0.6 0.70

REM Confidence 0.80
call run_noise_confidence_conda.bat 0.0 0.80
call run_noise_confidence_conda.bat 0.2 0.80
call run_noise_confidence_conda.bat 0.4 0.80
call run_noise_confidence_conda.bat 0.5 0.80
call run_noise_confidence_conda.bat 0.6 0.80

REM Confidence 0.90
call run_noise_confidence_conda.bat 0.0 0.90
call run_noise_confidence_conda.bat 0.2 0.90
call run_noise_confidence_conda.bat 0.4 0.90
call run_noise_confidence_conda.bat 0.5 0.90
call run_noise_confidence_conda.bat 0.6 0.90

REM Confidence 0.95
call run_noise_confidence_conda.bat 0.0 0.95
call run_noise_confidence_conda.bat 0.2 0.95
call run_noise_confidence_conda.bat 0.4 0.95
call run_noise_confidence_conda.bat 0.5 0.95
call run_noise_confidence_conda.bat 0.6 0.95

echo.
echo ============================================================================
echo FULL STUDY COMPLETE!
echo ============================================================================
echo.
echo Results:
echo   Transition: code\pnac_results\noise_0.45, noise_0.55, noise_0.60
echo   Confidence: code\pnac_results_confidence_study\conf_X.XX\noise_X.X\
echo.
pause
