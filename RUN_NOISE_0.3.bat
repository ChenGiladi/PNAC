@echo off
title PNAC - Noise Level 0.3 (Conda)
chcp 65001 >nul
cd /d "%~dp0"
call run_noise_conda.bat 0.3
pause
