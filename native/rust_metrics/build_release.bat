@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CARGO_BIN=%USERPROFILE%\.cargo\bin"
set "PATH=%CARGO_BIN%;%PATH%"

where cargo >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cargo not found. Install Rust toolchain first.
  exit /b 1
)

pushd "%SCRIPT_DIR%"
cargo build --release
if errorlevel 1 (
  popd
  echo [ERROR] Rust build failed.
  exit /b 1
)

set "OUT_DLL=%SCRIPT_DIR%target\release\lumina_metrics.dll"
if exist "%OUT_DLL%" (
  echo [OK] Built Rust metrics DLL: %OUT_DLL%
) else (
  echo [WARN] Expected DLL not found: %OUT_DLL%
)
popd
exit /b 0
