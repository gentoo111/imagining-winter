@echo off
rem S1-6: selected v3 (out\v3sel\cfgC_lr1e-3.pt) on 60 test seeds, DDIM 25, 50 paths -> out\marginal\v3.npz (+ clim, clock)
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\marginal mkdir D:\LastEmbers\e1\out\marginal
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\marg_run.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\marg_run.py eval v3
