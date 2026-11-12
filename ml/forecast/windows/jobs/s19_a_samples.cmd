@echo off
rem S1-9: v3 seed 1 on test seeds 0-59, 500 paths (50/200 = first-k subsets), DDIM 50,
rem 5-seed chunks until the segment budget; re-queues s19_a_samples.cmd until all chunks exist
rem -> out\samples\diff3s1_n500_endurance_murky.npz + done.json. Named to run after S1-8's s18_*. Code: windows\samples\ (SHA256SUMS).
set PYTHONUTF8=1
set DEVICE=cuda
set TORCH_THREADS=2
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\samples\parts mkdir D:\LastEmbers\e1\out\samples\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\samples.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\samples.py run --job s19_a_samples
