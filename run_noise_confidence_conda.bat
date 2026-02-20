@echo off
REM ============================================================================
REM Run PNAC experiments for a specific noise level AND confidence threshold
REM Usage: run_noise_confidence_conda.bat <noise_level> <confidence>
REM Example: run_noise_confidence_conda.bat 0.45 0.85
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

set NOISE_LEVEL=%1
set CONFIDENCE=%2

if "%NOISE_LEVEL%"=="" (
    echo Usage: %0 ^<noise_level^> ^<confidence^>
    echo Example: %0 0.45 0.85
    exit /b 1
)

if "%CONFIDENCE%"=="" (
    set CONFIDENCE=0.85
    echo Using default confidence: 0.85
)

title PNAC - Noise %NOISE_LEVEL% / Conf %CONFIDENCE%

echo ============================================================================
echo        PNAC EXPERIMENT - NOISE %NOISE_LEVEL% / CONFIDENCE %CONFIDENCE%
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
set OUTPUT_DIR=%CODE_DIR%\pnac_results_confidence_study

REM Configuration
set SEEDS=42 123 456
set EPOCHS=5
set ITERATIONS=5
set TTA=3
set BATCH_SIZE=160
set NUM_WORKERS=2

echo Configuration:
echo   Noise level: %NOISE_LEVEL%
echo   Confidence: %CONFIDENCE%
echo   Seeds: %SEEDS%
echo   Epochs: %EPOCHS%
echo   Iterations: %ITERATIONS%
echo   Data root: %DATA_ROOT%
echo.

cd /d "%CODE_DIR%"

for %%S in (%SEEDS%) do (
    set SEED=%%S
    set RUN_OUTPUT=%OUTPUT_DIR%\conf_%CONFIDENCE%\noise_%NOISE_LEVEL%\seed_%%S

    REM Check if already completed (search in timestamped subdirectories)
    set "FOUND_SUMMARY="
    for /d %%D in ("!RUN_OUTPUT!\pnac_*") do (
        if exist "%%D\summary.json" set "FOUND_SUMMARY=1"
    )
    if defined FOUND_SUMMARY (
        echo [SKIP] Already complete: noise=%NOISE_LEVEL%, conf=%CONFIDENCE%, seed=%%S
    ) else (
        echo ============================================================
        echo STARTING: noise=%NOISE_LEVEL%, conf=%CONFIDENCE%, seed=%%S
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
            echo ERROR: Experiment failed for noise=%NOISE_LEVEL%, conf=%CONFIDENCE%, seed=%%S
        ) else (
            echo Completed: noise=%NOISE_LEVEL%, conf=%CONFIDENCE%, seed=%%S
        )
        echo.
    )
)

echo.
echo ============================================================================
echo DONE! All experiments for noise=%NOISE_LEVEL%, conf=%CONFIDENCE% completed.
echo ============================================================================

endlocal
