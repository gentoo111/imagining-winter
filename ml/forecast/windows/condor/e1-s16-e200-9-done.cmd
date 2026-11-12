@echo off
rem HTCondor body of e1-s16-e200-9-done (legacy queue job s16_zz_e200_9_done.cmd; env from the job.toml).
rem S1-5: after the 4 model jobs -> out\eval200\done.json
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
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\eval200.py done
exit /b %ERRORLEVEL%
