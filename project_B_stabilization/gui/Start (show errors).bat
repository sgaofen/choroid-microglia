@echo off
rem Same as the normal launcher, but keeps a console window open to show errors.
cd /d "%~dp0"
"%~dp0python\python.exe" "%~dp0engine\app.py"
pause
