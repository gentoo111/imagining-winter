@echo off
rem S1-6 summary -> out\marginal\summary_endurance_murky.json, done.json
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\marginal mkdir D:\LastEmbers\e1\out\marginal
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\marg_run.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\marg_run.py summary
