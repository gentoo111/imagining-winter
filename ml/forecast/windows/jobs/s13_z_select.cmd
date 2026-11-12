@echo off
rem S1-3 v3 selection summary (after the 9 trainings): writes D:\LastEmbers\e1\out\v3sel\done.json
set PYTHONUTF8=1
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\v3_select.py select
