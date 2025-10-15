@echo off
echo 🚀 Dean Automation - Individual Stages
echo 📁 Directory: %CD%
echo.

REM Change to dean directory
cd dean

echo Choose which stage to run:
echo 1. Testing only
echo 2. Validation only  
echo 3. Scaling only
echo 4. All stages (default)
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo 🔄 Running Testing stage...
    python src/main.py --stage testing --profile production
) else if "%choice%"=="2" (
    echo 🔄 Running Validation stage...
    python src/main.py --stage validation --profile production
) else if "%choice%"=="3" (
    echo 🔄 Running Scaling stage...
    python src/main.py --stage scaling --profile production
) else (
    echo 🔄 Running All stages...
    python src/main.py --profile production
)

echo.
echo ✅ Dean automation completed
pause
