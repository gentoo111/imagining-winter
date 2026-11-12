@echo off
rem S1-4 summary (after the 3 seeds): writes D:\LastEmbers\e1\out\v3\done.json
set PYTHONUTF8=1
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\v3_seeds.py summary
