@echo off
echo 🚀 Setting up Dean Auto-Sync with GitHub...
echo.

REM Check if git is initialized
if not exist ".git" (
    echo ❌ Git repository not found. Please initialize git first:
    echo    git init
    echo    git remote add origin https://github.com/brava-skin/Dean
    echo    git add .
    echo    git commit -m "Initial commit"
    echo    git push -u origin main
    echo.
    pause
    exit /b 1
)

REM Check if remote origin exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo ❌ No remote origin found. Please add your GitHub repository:
    echo    git remote add origin https://github.com/brava-skin/Dean
    echo.
    pause
    exit /b 1
)

echo ✅ Git repository found
echo 📁 Current directory: %CD%
echo 🌐 Remote origin: 
git remote get-url origin
echo.

REM Create the auto-sync PowerShell script if it doesn't exist
if not exist "auto-sync.ps1" (
    echo ❌ auto-sync.ps1 not found. Please ensure the file exists.
    pause
    exit /b 1
)

REM Set execution policy for PowerShell (if needed)
echo 🔧 Setting up PowerShell execution policy...
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"

echo ✅ Auto-sync setup complete!
echo.
echo 🚀 To start auto-sync, run one of these commands:
echo.
echo   Option 1 - Simple: Double-click auto-sync.bat
echo   Option 2 - Manual: auto-sync.bat
echo   Option 3 - PowerShell: powershell -File auto-sync.ps1
echo.
echo 📝 The auto-sync will:
echo   - Monitor file changes every 30 seconds
echo   - Automatically commit changes
echo   - Push to GitHub
echo   - Show status messages
echo.
echo ⏹️  To stop auto-sync: Press Ctrl+C
echo.

REM Ask if user wants to start auto-sync now
set /p start_now="🤔 Start auto-sync now? (y/n): "
if /i "%start_now%"=="y" (
    echo 🚀 Starting auto-sync...
    call auto-sync.bat
) else (
    echo ✅ Setup complete! Run auto-sync.bat when ready.
    pause
)
