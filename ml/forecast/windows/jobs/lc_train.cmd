@echo off
rem S1-2 job 1: v2 diffusion 100k GPU steps, validation + checkpoint every 2k (~2.4 h, under the 3 h cap).
rem Writes D:\LastEmbers\e1\out\lc\done.json. console/locale is GBK; sources and balance.json are UTF-8
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\lc_train.py
