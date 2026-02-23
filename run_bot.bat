@echo off
setlocal
echo ---------------------------------------------------
echo [PRODUCTION] Starting Quants Agent in Resilient Mode
echo ---------------------------------------------------

:start
echo [INFO] Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo [INFO] Launching Live Trader at %TIME%...
python run_live.py >> logs\crash.log 2>&1

echo [WARNING] Bot crashed or stopped! Exit Code: %ERRORLEVEL%
echo [INFO] Restarting in 5 seconds...
timeout /t 5
goto start
