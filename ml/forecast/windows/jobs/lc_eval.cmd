@echo off
rem S1-2 job 2 (queued after lc_train): 60-seed BSS of the best checkpoint vs the paper's 12k model (minutes).
rem Writes D:\LastEmbers\e1\out\lc\summary.json. console/locale is GBK; sources and balance.json are UTF-8
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\lc_eval.py
