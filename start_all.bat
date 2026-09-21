@echo off
echo ====================================================
echo Starting VideoLens Full-Stack System...
echo ====================================================
start "VideoLens Backend" cmd /c "%~dp0run_backend.bat"
start "VideoLens Frontend" cmd /c "%~dp0run_frontend.bat"
echo Services launched!
echo Backend: http://127.0.0.1:8000/docs
echo Frontend: http://localhost:5173
pause
