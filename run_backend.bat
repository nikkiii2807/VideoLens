@echo off
echo ====================================================
echo Starting VideoLens Backend (FastAPI on port 8000)...
echo ====================================================
cd /d "%~dp0backend"
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin;%PATH%
"%~dp0backend\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
