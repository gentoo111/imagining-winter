@echo off
rem HTCondor body of e1-s1c-wn-a-train (legacy queue job s1c_wn_a_train.cmd; env from the job.toml).
rem S1-12b: diffusion retrained on the windnoise training set (selected-v3 recipe, 100000 steps); --then queue_eval writes nothing under Condor (e1-s1c-wn-b-eval follows via [after])
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
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py diff endurance murky_windnoise 100000 D:\LastEmbers\e1\out\windnoise\diff3_windnoise_s1.pt --job s1c_wn_a_train --done D:\LastEmbers\e1\out\windnoise\train_diff.json --then "windows\windnoise.py queue_eval"
exit /b %ERRORLEVEL%
