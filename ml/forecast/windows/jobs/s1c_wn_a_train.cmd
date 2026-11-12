@echo off
rem S1-12b GPU: diffusion retrained on the windnoise training set with the selected-v3 recipe (cfgC, lr 1e-3,
rem SEED 1, 100000 steps, TF32, VAL_EVERY 1000) chained through the queue; when done queues s1c_wn_b_eval.cmd.
rem Named to run after S1-5/6 (s16_*), S1-8 (s18_*) and S1-9 (s19_*).
set PYTHONUTF8=1
set PYTHONPATH=D:\LastEmbers\e1\ml\forecast
set E1_FC=D:\LastEmbers\e1\ml\forecast
set DEVICE=cuda
set SEED=1
set V3_CFG=C
set LR=1e-3
set TF32=1
set VAL_EVERY=1000
set CHAIN_TRAIN_PY=D:\LastEmbers\e1\ml\forecast\windows\windnoise\train.py
set CHAIN_ENV=V3_CFG,LR,TF32,SEED,E1_FC,PYTHONUTF8,CHAIN_ENV
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\windnoise\parts mkdir D:\LastEmbers\e1\out\windnoise\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\windnoise.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py diff endurance murky_windnoise 100000 D:\LastEmbers\e1\out\windnoise\diff3_windnoise_s1.pt --job s1c_wn_a_train --done D:\LastEmbers\e1\out\windnoise\train_diff.json --then "windows\windnoise.py queue_eval"
