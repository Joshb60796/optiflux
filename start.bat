@echo off
cd /d "%~dp0"
echo Starting OptiFlux desktop GUI...
python app.py
if errorlevel 1 (
  echo.
  echo If that failed, install deps:  pip install matplotlib numpy
  pause
)
