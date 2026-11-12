@echo off
rem HTCondor body of e1-s18-d-eval-newhome (legacy queue job s18_d_eval_newhome.cmd; env from the job.toml).
rem S1-8: a_new_home_murky v3, as s18_c -> out\secondary\diff3_a_new_home_murky.npz
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
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\secondary.py run a_new_home_murky --job s18_d_eval_newhome
exit /b %ERRORLEVEL%
