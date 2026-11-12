@echo off
rem S1-8: v3 cfgC lr 1e-3 seed 1 on endurance/clear, 100000 steps chained (train_chain.py re-queues s18_a_train_clear.cmd);
rem best-validation checkpoint D:\LastEmbers\e1\out\secondary\diff3_endurance_clear_s1.pt. Code: windows\secondary\train.py (SHA256SUMS).
set PYTHONUTF8=1
set DEVICE=cuda
set SEED=1
set V3_CFG=C
set LR=1e-3
set TF32=1
set VAL_EVERY=1000
set PYTHONPATH=D:\LastEmbers\e1\ml\forecast
set CHAIN_TRAIN_PY=D:\LastEmbers\e1\ml\forecast\windows\secondary\train.py
set CHAIN_ENV=V3_CFG,LR,TF32,SEED,CHAIN_ENV
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\secondary\runs mkdir D:\LastEmbers\e1\out\secondary\runs
if not exist D:\LastEmbers\e1\out\secondary\parts mkdir D:\LastEmbers\e1\out\secondary\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\secondary.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py diff endurance clear 100000 D:\LastEmbers\e1\out\secondary\diff3_endurance_clear_s1.pt --job s18_a_train_clear --done D:\LastEmbers\e1\out\secondary\runs\endurance_clear.json
