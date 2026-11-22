@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "MPLBACKEND=Agg"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 main.py %*
) else (
    python main.py %*
)
set "RUN_EXIT_CODE=%ERRORLEVEL%"
exit /b %RUN_EXIT_CODE%
