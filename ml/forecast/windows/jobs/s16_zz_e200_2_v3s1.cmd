@echo off
rem S1-5: v3s1 on test seeds 0-199, 200 paths, DDIM 50, 10-seed chunks until the segment budget;
rem re-queues s16_zz_e200_2_v3s1.cmd until all chunks exist -> out\eval200\diff3s1_endurance_murky.npz
rem Named to run right after S1-6's s16_* jobs. Code: windows\eval200\ (SHA256SUMS).
set PYTHONUTF8=1
set DEVICE=cuda
set TORCH_THREADS=2
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\eval200\parts mkdir D:\LastEmbers\e1\out\eval200\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py run v3s1 --job s16_zz_e200_2_v3s1
