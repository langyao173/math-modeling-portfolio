@echo off
cd /d "%~dp0"
py -3.10 scripts\run_kaiwu_batch.py --execute
if errorlevel 1 (
  echo Please check Python 3.10, project dependencies, and the locally licensed official Kaiwu SDK.
  echo Do not paste your SDK authorization code into a chat or report.
)
pause
