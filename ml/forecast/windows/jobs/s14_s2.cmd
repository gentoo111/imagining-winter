@echo off
rem S1-4 v3 training seed 2: cfgC, lr 1e-3, 100000 steps chained through the queue (train_chain.py
rem re-queues s14_s2.cmd until done). Writes D:\LastEmbers\e1\out\v3\diff3_endurance_murky_s2.pt(.log/.state/.chain.json), D:\LastEmbers\e1\out\v3\runs\s2.json.
rem console/locale is GBK; sources and balance.json are UTF-8
set PYTHONUTF8=1
set DEVICE=cuda
set SEED=2
set V3_CFG=C
set LR=1e-3
set TF32=1
set VAL_EVERY=1000
set CHAIN_ENV=V3_CFG,LR,TF32,SEED,CHAIN_ENV
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\v3\runs mkdir D:\LastEmbers\e1\out\v3\runs
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py diff endurance murky 100000 D:\LastEmbers\e1\out\v3\diff3_endurance_murky_s2.pt --job s14_s2 --done D:\LastEmbers\e1\out\v3\runs\s2.json
