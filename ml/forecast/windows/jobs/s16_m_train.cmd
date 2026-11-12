@echo off
rem S1-6 marginal control training: models.V3_CONFIGS['M'] (cfgC, diagonal attention, per-lead event channel),
rem lr 1e-3, 100000 steps chained through the queue (train_chain.py re-queues s16_m_train.cmd until done;
rem CHAIN_ENV lists itself, see E1-RESULTS S1-3). Code: windows\marg\train.py. Writes out\marginal\marg.pt(.log/.state/.chain.json)
set V3_CFG=M
set LR=1e-3
set TF32=1
set VAL_EVERY=1000
set PYTHONPATH=D:\LastEmbers\e1\ml\forecast
set CHAIN_TRAIN_PY=D:\LastEmbers\e1\ml\forecast\windows\marg\train.py
set CHAIN_ENV=V3_CFG,LR,TF32,CHAIN_ENV
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\marginal mkdir D:\LastEmbers\e1\out\marginal
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\marg_run.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py diff endurance murky 100000 D:\LastEmbers\e1\out\marginal\marg.pt --job s16_m_train --done D:\LastEmbers\e1\out\marginal\train_done.json
