@echo off
echo 🚀 Dean Auto-Sync
echo 📁 Directory: %CD%
echo 🌐 GitHub: https://github.com/brava-skin/Dean
echo.

REM Set execution policy
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force" >nul 2>&1

REM Run the sync script
powershell -ExecutionPolicy Bypass -File "sync.ps1"

pause
