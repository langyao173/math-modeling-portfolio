@echo off
setlocal
cd /d "%~dp0"
python build_latex.py %*
exit /b %ERRORLEVEL%
