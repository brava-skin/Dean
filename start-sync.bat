@echo off
echo 🚀 Starting Dean Auto-Sync...
echo 📁 Directory: %CD%
echo 🌐 GitHub: https://github.com/brava-skin/Dean
echo ⏱️  Auto-sync every 30 seconds
echo.

REM Check if PowerShell execution policy allows scripts
powershell -Command "Get-ExecutionPolicy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  PowerShell execution policy issue detected
    echo 🔧 Fixing execution policy...
    powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"
    echo ✅ Execution policy updated
    echo.
)

echo 🚀 Starting auto-sync...
powershell -ExecutionPolicy Bypass -File "simple-sync.ps1" -GitHubRepo "https://github.com/brava-skin/Dean.git" -IntervalSeconds 30

pause
