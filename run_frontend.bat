@echo off
echo ====================================================
echo Starting VideoLens Frontend (Vite on port 5173)...
echo ====================================================
cd /d "%~dp0frontend"
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v24.19.0-win-arm64;%PATH%
npm run dev
pause
