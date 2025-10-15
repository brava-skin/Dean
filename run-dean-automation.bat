@echo off
echo 🚀 Starting Dean Automation...
echo 📁 Directory: %CD%
echo ⏰ Time: %date% %time%
echo.

REM Change to dean directory
cd dean

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run Dean automation
echo 🔄 Running Dean automation (all stages)...
python src/main.py --profile production

echo.
echo ✅ Dean automation completed
echo ⏰ Finished at: %date% %time%
