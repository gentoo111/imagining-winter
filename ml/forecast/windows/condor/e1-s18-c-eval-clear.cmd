@echo off
rem HTCondor body of e1-s18-c-eval-clear (legacy queue job s18_c_eval_clear.cmd; env from the job.toml).
rem S1-8: endurance_clear v3 on test seeds 0-199, 200 paths, DDIM 50 (as S1-5) -> out\secondary\diff3_endurance_clear.npz
rem Match the legacy queue: it never set these, the Condor wrapper sets them to the slot's CPUs.
set OMP_NUM_THREADS=
set MKL_NUM_THREADS=
set OPENBLAS_NUM_THREADS=
set TORCH_NUM_THREADS=
set RAYON_NUM_THREADS=
cd /d D:\LastEmbers\e1\ml\forecast
if not exist D:\LastEmbers\e1\out\secondary\runs mkdir D:\LastEmbers\e1\out\secondary\runs
if not exist D:\LastEmbers\e1\out\secondary\parts mkdir D:\LastEmbers\e1\out\secondary\parts
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\secondary.py verify
if errorlevel 1 exit /b 1
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\secondary.py run endurance_clear --job s18_c_eval_clear
exit /b %ERRORLEVEL%
