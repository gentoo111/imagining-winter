@echo off
rem HTCondor body of e1-s19-a-samples (legacy queue job s19_a_samples.cmd; env from the job.toml).
rem S1-9: v3 seed 1 on test seeds 0-59, 500 paths (50/200 = first-k subsets), DDIM 50, 5-seed chunks -> out\samples\diff3s1_n500_endurance_murky.npz + done.json
rem Match the legacy queue: it never set these, the Condor wrapper sets them to the slot's CPUs.
set OMP_NUM_THREADS=
set MKL_NUM_THREADS=
set OPENBLAS_NUM_THREADS=
set TORCH_NUM_THREADS=
set RAYON_NUM_THREADS=
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\samples\parts mkdir D:\LastEmbers\e1\out\samples\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\samples.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\samples.py run --job s19_a_samples
exit /b %ERRORLEVEL%
