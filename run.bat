@echo off
setlocal
cd /d "%~dp0"

REM The GUI needs Python 3.12 — PyQt5 has no wheels for 3.7, which is what a
REM bare "python" was picking up off PATH and caused the "DLL load failed"
REM crash.  Force 3.12 through the Windows Python launcher, same as train.bat.
set "PY=py -3.12"

%PY% --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Python 3.12 was not found via the "py" launcher.
    echo   Install it from https://www.python.org/downloads/ ^(tick "py launcher"^)
    echo   and run this again.
    echo.
    pause
    exit /b 1
)

REM First-time setup: if PyQt5 isn't installed for 3.12, pull in the core deps.
%PY% -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo First run: installing dependencies for Python 3.12 . . .
    %PY% -m pip install -r requirements-core.txt
)

%PY% main.py
pause
