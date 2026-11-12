@echo off
rem S1-1b GPU smoke + speed measurement (minutes). Writes D:\LastEmbers\e1\out\smoke\done.json.
rem console/locale is GBK; sources and balance.json are UTF-8
set PYTHONUTF8=1
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\smoke.py
