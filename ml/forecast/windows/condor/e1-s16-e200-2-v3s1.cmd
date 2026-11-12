@echo off
rem HTCondor body of e1-s16-e200-2-v3s1 (legacy queue job s16_zz_e200_2_v3s1.cmd; env from the job.toml).
rem S1-5: v3s1 on test seeds 0-199, 200 paths, DDIM 50, 10-seed chunks -> out\eval200\diff3s1_endurance_murky.npz. Resumes from the parts already on disk (v3s1_000..050 written by the legacy queue).
rem Match the legacy queue: it never set these, the Condor wrapper sets them to the slot's CPUs.
set OMP_NUM_THREADS=
set MKL_NUM_THREADS=
set OPENBLAS_NUM_THREADS=
set TORCH_NUM_THREADS=
set RAYON_NUM_THREADS=
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\eval200\parts mkdir D:\LastEmbers\e1\out\eval200\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py run v3s1 --job s16_zz_e200_2_v3s1
exit /b %ERRORLEVEL%
