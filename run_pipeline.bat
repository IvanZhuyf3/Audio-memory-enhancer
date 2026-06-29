@echo off
REM Audio-memory-enhancer pipeline launcher.
REM Activates the SHARED venv (audio_transcribe) and runs pipeline.py with all args.
REM Use this as the Windows Task Scheduler action target:
REM   Program: C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat
REM   Arguments: sync

setlocal
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

set VENV=C:\Users\Yifan\venvs\audio_transcribe\Scripts\activate.bat
set SCRIPT_DIR=%~dp0

if not exist "%VENV%" (
    echo [ERROR] Shared venv not found at %VENV%
    echo         This project reuses the audio_transcribe venv. Do not create a new one.
    exit /b 1
)

call "%VENV%"
if errorlevel 1 (
    echo [ERROR] Failed to activate venv.
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
python pipeline.py %*
exit /b %ERRORLEVEL%
