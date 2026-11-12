@echo off
rem S1-3 v3 speed check (~15 min): 600 steps of v2 and v3 candidates A..C, TF32 off/on.
rem Writes D:\LastEmbers\e1\out\v3sel\bench.json. console/locale is GBK; sources and balance.json are UTF-8
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\v3_bench.py
