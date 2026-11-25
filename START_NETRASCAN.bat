@echo off
setlocal
title NetraScan.AI V2 - Application Launcher

cd /d "%~dp0"

set "MPLBACKEND=Agg"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

echo ==========================================================
echo                    NetraScan.AI V2
echo              Three Transformers. One Vision.
echo ==========================================================
echo.
echo Multi-Transformer Classification
echo Grad-CAM Explainability
echo Local Generative AI
echo Automated PDF Research Reports
echo.
echo Project: %CD%
echo.

if not exist "app.py" (
    echo ERROR: Application source not found.
    goto :failure
)

if not exist "llm\manifest.json" (
    echo ERROR: Local LLM manifest not found.
    goto :failure
)

if not exist "models\vit_v4\vit_v4_best.pth" (
    echo ERROR: ViT-B/16 checkpoint not found.
    goto :missing_models
)

if not exist "models\swin_v4\swin_v4_best.pth" (
    echo ERROR: Swin-Tiny checkpoint not found.
    goto :missing_models
)

if not exist "models\mobilevit_v4\mobilevit_v4_best.pth" (
    echo ERROR: MobileViT-S checkpoint not found.
    goto :missing_models
)

if exist "runtime\python\python.exe" (
    set "PYTHON_EXE=runtime\python\python.exe"
    echo Environment: Portable Python runtime
    goto :environment_ready
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    echo Environment: Local virtual environment
    goto :environment_ready
)

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python runtime not found.
    echo Follow the installation instructions in README.md.
    goto :failure
)

set "PYTHON_EXE=python"
echo Environment: Python available on PATH

:environment_ready

if not exist "llm\backend\llama-server.exe" (
    echo WARNING: Local LLM backend not found.
    echo AI narrative generation may use deterministic fallback.
)

if not exist "llm\weights\qwen2.5-3b-instruct-q4_k_m.gguf" (
    echo WARNING: Local LLM weights not found.
    echo AI narrative generation may use deterministic fallback.
)

echo.
echo Python executable: %PYTHON_EXE%
echo Starting NetraScan.AI V2...
echo.
echo Local application:
echo http://127.0.0.1:7860
echo.
echo Keep this terminal open while using the application.
echo Use Stop Application in the interface to exit.
echo Use Release Local LLM Resources to unload the LLM.
echo.
echo Academic research prototype - Not for clinical diagnosis.
echo.

"%PYTHON_EXE%" -B -u "app.py"

set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo NetraScan.AI V2 has stopped.
echo Exit code: %EXIT_CODE%
pause

endlocal & exit /b %EXIT_CODE%

:missing_models
echo.
echo Vision checkpoints are not bundled with the GitHub repository.
echo Provision the required checkpoints as described in README.md.

:failure
echo.
pause
endlocal
exit /b 1
