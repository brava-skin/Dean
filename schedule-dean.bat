@echo off
echo 🕐 Dean Automation Scheduler
echo 📁 Directory: %CD%
echo ⏰ Starting at: %date% %time%
echo.

REM Set interval (in minutes) - default 2 hours = 120 minutes
set INTERVAL_MINUTES=120

echo ⏱️  Running Dean automation every %INTERVAL_MINUTES% minutes
echo 🛑 Press Ctrl+C to stop
echo.

:loop
echo.
echo 🔄 Running Dean automation at %date% %time%
call run-dean-automation.bat

echo.
echo ⏳ Waiting %INTERVAL_MINUTES% minutes until next run...
timeout /t %INTERVAL_MINUTES% /nobreak >nul

goto loop
