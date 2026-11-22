@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "MPLBACKEND=Agg"
python build_report.py
exit /b %ERRORLEVEL%
