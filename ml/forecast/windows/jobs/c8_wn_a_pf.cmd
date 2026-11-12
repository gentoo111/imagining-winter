@echo off
rem S1-12b CPU lane: climatology + PF oracle (flip-aware wind likelihood) on the 60 windnoise test seeds
rem -> out\windnoise\{clim,pf}_endurance_murky_windnoise.npz. Code: windows\windnoise\ (SHA256SUMS).
set PYTHONUTF8=1
set PYTHONPATH=D:\LastEmbers\e1\ml\forecast
set E1_FC=D:\LastEmbers\e1\ml\forecast
set CUDA_VISIBLE_DEVICES=-1
set DEVICE=cpu
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set TORCH_THREADS=4
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\windnoise\parts mkdir D:\LastEmbers\e1\out\windnoise\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\windnoise.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\windnoise.py pf
