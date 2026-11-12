@echo off
rem S1-5 CPU lane: path baselines (clim / clock / GRU + analogue paths) on the 200 test seeds,
rem 200 paths per forecast -> out\eval200\pathbase_endurance_murky_200.json (CPU only)
set PYTHONUTF8=1
set CUDA_VISIBLE_DEVICES=-1
set DEVICE=cpu
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set TORCH_THREADS=4
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\eval200\parts mkdir D:\LastEmbers\e1\out\eval200\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py pathbase
