@echo off
REM ============================================================================
REM PNAC Conda Environment Setup with CUDA
REM ============================================================================
REM This script creates a conda environment with PyTorch + CUDA for PNAC
REM Run this ONCE before running experiments
REM ============================================================================

title PNAC Conda Setup (CUDA)
chcp 65001 >nul

echo ============================================================================
echo        PNAC CONDA ENVIRONMENT SETUP (WITH CUDA)
echo ============================================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: Conda not found in PATH.
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo Conda found. Creating environment...
echo.

REM Remove existing environment if it exists
echo [1/5] Removing existing 'pnac' environment if present...
call conda deactivate 2>nul
call conda env remove -n pnac -y 2>nul

REM Create conda environment with Python 3.11
echo.
echo [2/5] Creating conda environment 'pnac' with Python 3.11...
call conda create -n pnac python=3.11 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

echo.
echo [3/5] Activating environment...
call conda activate pnac
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    pause
    exit /b 1
)

echo.
echo [4/5] Installing PyTorch with CUDA 12.1 support...
echo Using pip for reliable CUDA installation...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch with CUDA.
    echo Trying alternative installation method...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo ERROR: Failed to install PyTorch.
        pause
        exit /b 1
    )
)

echo.
echo [5/5] Installing additional dependencies...
pip install numpy pillow matplotlib scipy tqdm

echo.
echo ============================================================================
echo Verifying CUDA installation...
echo ============================================================================
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo ============================================================================
    echo WARNING: CUDA is NOT available!
    echo ============================================================================
    echo PyTorch installed but CUDA not detected.
    echo Make sure you have:
    echo   1. NVIDIA GPU with CUDA support
    echo   2. Latest NVIDIA drivers installed
    echo.
    echo The experiments will run on CPU (slower).
    echo.
) else (
    echo.
    echo ============================================================================
    echo SUCCESS! CUDA is available.
    echo ============================================================================
)

echo.
echo Conda environment 'pnac' is ready.
echo You can now run the RUN_NOISE_0.X.bat scripts.
echo.
pause
