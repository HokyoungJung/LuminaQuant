@echo off
setlocal enabledelayedexpansion
echo ---------------------------------------------------
echo [PRODUCTION] Starting LuminaQuant in Resilient Mode
echo ---------------------------------------------------

if not exist logs mkdir logs

REM The RotatingFileHandler already captures structured logs to
REM logs\lumina_quant.log; skip the stderr console handler so it is not
REM duplicated into the redirected crash.log below (audit O5).
set LQ_DISABLE_CONSOLE_LOG=1

:start
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] uv is required but was not found in PATH.
  exit /b 1
)

call :rotate_crash_log

echo [INFO] Launching Live Trader at %TIME%...
uv run lq live >> logs\crash.log 2>&1
set "EXITCODE=%ERRORLEVEL%"

REM Mirror run_bot.sh's restart policy (audit M1/O5): only restart on an
REM unexpected crash. A clean exit (0), an operator interrupt (130/143, the
REM Windows-reachable analog of run_bot.sh's SIGINT/SIGTERM check), or the
REM fail-fast / kill-switch hard-halt exit (2, raised by cli/live.py's
REM _shutdown_on_fatal) must NOT be auto-restarted -- restarting would re-arm
REM trading with a zeroed consecutive-loss counter and defeat the kill-switch.
if %EXITCODE% EQU 0 (
  echo [INFO] Live Trader exited cleanly. Not restarting.
  exit /b 0
)

if %EXITCODE% EQU 130 (
  echo [INFO] Live Trader was interrupted ^(Ctrl+C^). Not restarting.
  exit /b 130
)

if %EXITCODE% EQU 143 (
  echo [INFO] Live Trader was terminated. Not restarting.
  exit /b 143
)

if %EXITCODE% EQU 2 (
  echo [ERROR] Live Trader hit a fail-fast / kill-switch hard-halt ^(exit code 2^).
  echo [ERROR] Not restarting -- operator review is required before relaunch.
  exit /b 2
)

echo [WARNING] Bot crashed! Exit Code: %EXITCODE%
echo [INFO] Restarting in 5 seconds...
timeout /t 5
goto start

REM Bound crash.log so it cannot grow without limit across restarts.
REM Keeps up to 5 ~10 MiB backups, mirroring the RotatingFileHandler policy.
:rotate_crash_log
if not exist "logs\crash.log" goto :eof
set "CRASHSIZE=0"
for %%A in ("logs\crash.log") do set "CRASHSIZE=%%~zA"
if !CRASHSIZE! LSS 10485760 goto :eof
if exist "logs\crash.log.5" del /q "logs\crash.log.5"
if exist "logs\crash.log.4" move /y "logs\crash.log.4" "logs\crash.log.5" >nul
if exist "logs\crash.log.3" move /y "logs\crash.log.3" "logs\crash.log.4" >nul
if exist "logs\crash.log.2" move /y "logs\crash.log.2" "logs\crash.log.3" >nul
if exist "logs\crash.log.1" move /y "logs\crash.log.1" "logs\crash.log.2" >nul
move /y "logs\crash.log" "logs\crash.log.1" >nul
goto :eof
