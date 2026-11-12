@echo off
rem HTCondor body of e1-s16-e200-4-v3s3 (legacy queue job s16_zz_e200_4_v3s3.cmd; env from the job.toml).
rem S1-5: v3s3, as v3s1 -> out\eval200\diff3s3_endurance_murky.npz
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
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py run v3s3 --job s16_zz_e200_4_v3s3
exit /b %ERRORLEVEL%
