@echo off
REM ============================================================================
REM Run PNAC experiments for a specific noise level using Conda
REM Usage: run_noise_conda.bat <noise_level>
REM Example: run_noise_conda.bat 0.1
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

set NOISE_LEVEL=%1

if "%NOISE_LEVEL%"=="" (
    echo Usage: %0 ^<noise_level^>
    echo Example: %0 0.1
    exit /b 1
)

title PNAC - Noise Level %NOISE_LEVEL%

echo ============================================================================
echo        PNAC EXPERIMENT - NOISE LEVEL %NOISE_LEVEL%
echo ============================================================================
echo.

REM Activate conda environment
call conda activate pnac
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'pnac'.
    echo Please run SETUP_CONDA.bat first.
    pause
    exit /b 1
)

REM Set paths
set SCRIPT_DIR=%~dp0
set CODE_DIR=%SCRIPT_DIR%code
set DATA_ROOT=%SCRIPT_DIR%data\three_classes_dataset
set MANIFEST=%SCRIPT_DIR%data\three_classes_split_manifest.json
set OUTPUT_DIR=%CODE_DIR%\pnac_results

REM Configuration - Optimized for speed and cascade effect
set SEEDS=42 123 456
set EPOCHS=5
set ITERATIONS=5
set CONFIDENCE=0.85
set TTA=3
set BATCH_SIZE=160
set NUM_WORKERS=2

echo Configuration:
echo   Noise level: %NOISE_LEVEL%
echo   Seeds: %SEEDS%
echo   Epochs: %EPOCHS%
echo   Iterations: %ITERATIONS%
echo   Data root: %DATA_ROOT%
echo.

cd /d "%CODE_DIR%"

for %%S in (%SEEDS%) do (
    set SEED=%%S
    set RUN_OUTPUT=%OUTPUT_DIR%\noise_%NOISE_LEVEL%\seed_%%S

    REM Check if already completed (search in timestamped subdirectories)
    set "FOUND_SUMMARY="
    for /d %%D in ("!RUN_OUTPUT!\pnac_*") do (
        if exist "%%D\summary.json" set "FOUND_SUMMARY=1"
    )
    if defined FOUND_SUMMARY (
        echo [SKIP] Already complete: noise=%NOISE_LEVEL%, seed=%%S
    ) else (
        echo ============================================================
        echo STARTING: noise=%NOISE_LEVEL%, seed=%%S
        echo ============================================================

        if not exist "!RUN_OUTPUT!" mkdir "!RUN_OUTPUT!"

        python pnac.py ^
            --data-root "%DATA_ROOT%" ^
            --manifest "%MANIFEST%" ^
            --output-dir "!RUN_OUTPUT!" ^
            --noise-rate %NOISE_LEVEL% ^
            --iterations %ITERATIONS% ^
            --epochs %EPOCHS% ^
            --confidence %CONFIDENCE% ^
            --tta %TTA% ^
            --seed %%S ^
            --batch-size %BATCH_SIZE% ^
            --num-workers %NUM_WORKERS% ^
            --pretrained

        if errorlevel 1 (
            echo ERROR: Experiment failed for noise=%NOISE_LEVEL%, seed=%%S
        ) else (
            echo Completed: noise=%NOISE_LEVEL%, seed=%%S
        )
        echo.
    )
)

echo.
echo ============================================================================
echo DONE! All experiments for noise level %NOISE_LEVEL% completed.
echo ============================================================================

endlocal
