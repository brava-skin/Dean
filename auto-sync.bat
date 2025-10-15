@echo off
REM Auto-sync batch file for Windows
REM This script starts the PowerShell auto-sync with default settings

echo 🚀 Starting Dean Auto-Sync...
echo 📁 Directory: %CD%
echo 🌐 GitHub: Auto-pushing changes every 30 seconds
echo ⏹️  Press Ctrl+C to stop
echo.

REM Start PowerShell auto-sync script
powershell -ExecutionPolicy Bypass -File "auto-sync.ps1" -Branch main -IntervalSeconds 30 -Verbose

pause
