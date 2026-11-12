@echo off
rem S1-8: endurance_clear v3 on test seeds 0-199, 200 paths, DDIM 50 (as S1-5), 10-seed chunks;
rem re-queues s18_c_eval_clear.cmd until all chunks exist -> D:\LastEmbers\e1\out\secondary\diff3_endurance_clear.npz
set PYTHONUTF8=1
set DEVICE=cuda
set TORCH_THREADS=2
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\secondary\runs mkdir D:\LastEmbers\e1\out\secondary\runs
if not exist D:\LastEmbers\e1\out\secondary\parts mkdir D:\LastEmbers\e1\out\secondary\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\secondary.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\secondary.py run endurance_clear --job s18_c_eval_clear
