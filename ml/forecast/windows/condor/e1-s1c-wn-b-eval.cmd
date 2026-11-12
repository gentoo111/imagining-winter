@echo off
rem HTCondor body of e1-s1c-wn-b-eval (legacy queue job s1c_wn_b_eval.cmd; env from the job.toml).
rem S1-12b: the windnoise-trained v3 on the 60 windnoise test seeds, 200 paths, DDIM 50, 10-seed chunks; then merge + mark diff (summary, all three parts done)
rem Match the legacy queue: it never set these, the Condor wrapper sets them to the slot's CPUs.
set OMP_NUM_THREADS=
set MKL_NUM_THREADS=
set OPENBLAS_NUM_THREADS=
set TORCH_NUM_THREADS=
set RAYON_NUM_THREADS=
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\windnoise\parts mkdir D:\LastEmbers\e1\out\windnoise\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\windnoise.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\windnoise.py diff --job s1c_wn_b_eval
exit /b %ERRORLEVEL%
