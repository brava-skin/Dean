@echo off
echo 🚀 Dean Auto-Sync
echo 📁 Directory: %CD%
echo 🌐 GitHub: https://github.com/brava-skin/Dean
echo ⏱️  Checking every 30 seconds
echo.

REM Check if Git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git not found. Please install Git first:
    echo    1. Download from: https://git-scm.com/download/win
    echo    2. Install with default settings
    echo    3. Restart terminal and try again
    pause
    exit /b 1
)

echo ✅ Git found
echo.

REM Check if this is a git repository
if not exist ".git" (
    echo ⚠️  Initializing Git repository...
    git init
    git remote add origin https://github.com/brava-skin/Dean.git
    git add .
    git commit -m "Initial commit"
    git branch -M main
    echo ✅ Git repository initialized!
    echo.
)

echo 🚀 Starting auto-sync monitoring...
echo Press Ctrl+C to stop
echo.

:loop
REM Check for changes
git status --porcelain >nul 2>&1
if errorlevel 1 (
    echo 📝 No changes detected at %time%
) else (
    echo 🔄 Changes detected, syncing...
    
    REM Add all changes
    git add .
    
    REM Get timestamp
    for /f "tokens=1-3 delims= " %%a in ('date /t') do set mydate=%%c-%%a-%%b
    for /f "tokens=1-2 delims=:" %%a in ('time /t') do set mytime=%%a:%%b
    set timestamp=%mydate% %mytime%
    
    REM Commit and push
    git commit -m "🔄 Auto-sync: %timestamp%"
    if errorlevel 0 (
        git push origin main
        if errorlevel 0 (
            echo ✅ Successfully synced to GitHub!
        ) else (
            echo ❌ Failed to push to GitHub
        )
    ) else (
        echo ⚠️  No changes to commit
    )
)

REM Wait 30 seconds
timeout /t 30 /nobreak >nul
goto loop
