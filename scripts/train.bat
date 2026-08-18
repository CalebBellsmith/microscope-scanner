@echo off
cd /d "%~dp0..\app"
py -3.12 labeling_tool.py
pause
