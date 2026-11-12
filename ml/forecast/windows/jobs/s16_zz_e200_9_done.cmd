@echo off
rem S1-5: after the 4 model jobs -> out\eval200\done.json
set PYTHONUTF8=1
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\eval200\parts mkdir D:\LastEmbers\e1\out\eval200\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py done
