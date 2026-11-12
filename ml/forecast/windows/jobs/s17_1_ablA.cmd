@echo off
rem S1-7 ablation job 1/4 (variant A, see models.VARIANTS): 12k GPU steps (~17 min) + 60-seed eval (DDIM 25, 50 paths).
rem Writes D:\LastEmbers\e1\out\ablation\{ablA.pt, summary_ablA.json, done.json}. console/locale is GBK; sources and balance.json are UTF-8
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\abl_run.py A
