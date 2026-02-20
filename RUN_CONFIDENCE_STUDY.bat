@echo off
REM ============================================================================
REM CONFIDENCE THRESHOLD STUDY
REM
REM Purpose: Show how confidence threshold τ affects PNAC sensitivity
REM          Lower τ → more pseudo-labels → potentially earlier degradation
REM          Higher τ → fewer pseudo-labels → more robust but less sensitive
REM
REM Matrix:
REM   Noise levels: 0.0, 0.2, 0.4, 0.5, 0.6
REM   Confidence:   0.70, 0.80, 0.90, 0.95
REM
REM Runs: 5 noise × 4 confidence × 3 seeds = 60 experiments
REM ============================================================================

title PNAC - Confidence Threshold Study
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================================
echo        PNAC CONFIDENCE THRESHOLD STUDY
echo ============================================================================
echo.
echo This study varies confidence threshold to show sensitivity:
echo   Confidence thresholds: 0.70, 0.80, 0.90, 0.95
echo   Noise levels: 0.0, 0.2, 0.4, 0.5, 0.6
echo.
echo Total experiments: 5 noise × 4 confidence × 3 seeds = 60 runs
echo Expected runtime: ~5-6 hours
echo.
echo Results will be saved to: code\pnac_results_confidence_study\
echo.
echo Press any key to start or Ctrl+C to cancel...
pause >nul

REM ============================================================================
REM CONFIDENCE = 0.70 (Most permissive - expects earliest degradation)
REM ============================================================================
echo.
echo ========================================
echo CONFIDENCE THRESHOLD: 0.70
echo ========================================

call run_noise_confidence_conda.bat 0.0 0.70
call run_noise_confidence_conda.bat 0.2 0.70
call run_noise_confidence_conda.bat 0.4 0.70
call run_noise_confidence_conda.bat 0.5 0.70
call run_noise_confidence_conda.bat 0.6 0.70

REM ============================================================================
REM CONFIDENCE = 0.80
REM ============================================================================
echo.
echo ========================================
echo CONFIDENCE THRESHOLD: 0.80
echo ========================================

call run_noise_confidence_conda.bat 0.0 0.80
call run_noise_confidence_conda.bat 0.2 0.80
call run_noise_confidence_conda.bat 0.4 0.80
call run_noise_confidence_conda.bat 0.5 0.80
call run_noise_confidence_conda.bat 0.6 0.80

REM ============================================================================
REM CONFIDENCE = 0.90
REM ============================================================================
echo.
echo ========================================
echo CONFIDENCE THRESHOLD: 0.90
echo ========================================

call run_noise_confidence_conda.bat 0.0 0.90
call run_noise_confidence_conda.bat 0.2 0.90
call run_noise_confidence_conda.bat 0.4 0.90
call run_noise_confidence_conda.bat 0.5 0.90
call run_noise_confidence_conda.bat 0.6 0.90

REM ============================================================================
REM CONFIDENCE = 0.95 (Most restrictive - expects latest degradation)
REM ============================================================================
echo.
echo ========================================
echo CONFIDENCE THRESHOLD: 0.95
echo ========================================

call run_noise_confidence_conda.bat 0.0 0.95
call run_noise_confidence_conda.bat 0.2 0.95
call run_noise_confidence_conda.bat 0.4 0.95
call run_noise_confidence_conda.bat 0.5 0.95
call run_noise_confidence_conda.bat 0.6 0.95

echo.
echo ============================================================================
echo CONFIDENCE STUDY COMPLETE!
echo ============================================================================
echo.
echo Results saved to: code\pnac_results_confidence_study\
echo   conf_0.70\noise_X.X\seed_Y\
echo   conf_0.80\noise_X.X\seed_Y\
echo   conf_0.90\noise_X.X\seed_Y\
echo   conf_0.95\noise_X.X\seed_Y\
echo.
echo Expected findings:
echo   - Lower confidence (0.70): Earlier F1 degradation, more sensitive
echo   - Higher confidence (0.95): Later F1 degradation, more robust
echo.
pause
