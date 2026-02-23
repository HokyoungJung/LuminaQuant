@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SRC=%SCRIPT_DIR%evaluate_metrics.c"
set "OUT_DIR=%SCRIPT_DIR%build"
set "OUT_DLL=%OUT_DIR%\lumina_metrics.dll"

if not exist "%SRC%" (
  echo [ERROR] Source not found: %SRC%
  exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

if "%VSINSTALLDIR%"=="" (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
      set "VSINSTALLDIR=%%i"
    )
  )
)

if "%VSINSTALLDIR%"=="" (
  echo [ERROR] Visual Studio Build Tools not found.
  exit /b 1
)

set "VSDEVCMD=%VSINSTALLDIR%\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
  echo [ERROR] VsDevCmd not found: %VSDEVCMD%
  exit /b 1
)

call "%VSDEVCMD%" -no_logo -arch=x64 -host_arch=x64
if errorlevel 1 (
  echo [ERROR] Failed to initialize MSVC environment.
  exit /b 1
)

cl /nologo /O2 /LD "%SRC%" /link /OUT:"%OUT_DLL%"
if errorlevel 1 (
  echo [ERROR] MSVC build failed.
  exit /b 1
)

echo [OK] Built native metrics DLL: %OUT_DLL%
exit /b 0
