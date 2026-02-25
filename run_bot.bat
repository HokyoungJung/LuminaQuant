@echo off
setlocal
echo ---------------------------------------------------
echo [PRODUCTION] Starting Quants Agent in Resilient Mode
echo ---------------------------------------------------

:start
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] uv is required but was not found in PATH.
  exit /b 1
)

echo [INFO] Launching Live Trader at %TIME%...
uv run python run_live.py >> logs\crash.log 2>&1

echo [WARNING] Bot crashed or stopped! Exit Code: %ERRORLEVEL%
echo [INFO] Restarting in 5 seconds...
timeout /t 5
goto start
