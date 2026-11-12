@echo off
rem S1-3 v3 selection: config B (models.V3_CONFIGS), peak lr 1e-4, 100000 steps chained through the queue
rem (train_chain.py re-queues s13_B1_lr1e-4.cmd until done). Writes D:\LastEmbers\e1\out\v3sel\cfgB_lr1e-4.pt(.log/.state/.chain.json),
rem D:\LastEmbers\e1\out\v3sel\runs\cfgB_lr1e-4.json. console/locale is GBK; sources and balance.json are UTF-8
set PYTHONUTF8=1
set DEVICE=cuda
set V3_CFG=B
set LR=1e-4
set TF32=1
set VAL_EVERY=1000
set CHAIN_ENV=V3_CFG,LR,TF32,CHAIN_ENV
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\v3sel\runs mkdir D:\LastEmbers\e1\out\v3sel\runs
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py diff endurance murky 100000 D:\LastEmbers\e1\out\v3sel\cfgB_lr1e-4.pt --job s13_B1_lr1e-4 --done D:\LastEmbers\e1\out\v3sel\runs\cfgB_lr1e-4.json
