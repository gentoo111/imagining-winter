@echo off
rem S1-3a GPU resume smoke (minutes): v2 diff 400 steps once vs chained (train_chain.py time budget -> re-queued
rem s3a_resume_smoke_chain.cmd). Writes D:\LastEmbers\e1\out\resume_smoke\done.json. console/locale is GBK; sources UTF-8
set PYTHONUTF8=1
set DEVICE=cuda
cd /d D:\LastEmbers\e1\ml\forecast
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\resume_smoke.py start
