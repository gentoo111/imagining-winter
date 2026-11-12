@echo off
rem S1-12b CPU lane: GRU retrained on the windnoise training set, 16000 steps (main-table recipe), chained
rem through queue-cpu (train_chain.py re-queues c8_wn_b_gru.cmd), then evaluated on the 60 test seeds.
set PYTHONUTF8=1
set PYTHONPATH=D:\LastEmbers\e1\ml\forecast
set E1_FC=D:\LastEmbers\e1\ml\forecast
set CUDA_VISIBLE_DEVICES=-1
set DEVICE=cpu
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set TORCH_THREADS=4
set CHAIN_QUEUE=D:\LastEmbers\e1\queue-cpu
set CHAIN_TRAIN_PY=D:\LastEmbers\e1\ml\forecast\windows\windnoise\train.py
set CHAIN_ENV=CUDA_VISIBLE_DEVICES,OMP_NUM_THREADS,MKL_NUM_THREADS,OPENBLAS_NUM_THREADS,E1_FC,PYTHONUTF8,CHAIN_ENV
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\windnoise\parts mkdir D:\LastEmbers\e1\out\windnoise\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\windnoise.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py gru endurance murky_windnoise 16000 D:\LastEmbers\e1\out\windnoise\gru16k_windnoise.pt --job c8_wn_b_gru --done D:\LastEmbers\e1\out\windnoise\train_gru.json --then "windows\windnoise.py gru_eval"
