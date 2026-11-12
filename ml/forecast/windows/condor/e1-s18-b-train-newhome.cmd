@echo off
rem HTCondor body of e1-s18-b-train-newhome (legacy queue job s18_b_train_newhome.cmd; env from the job.toml).
rem S1-8: as s18_a on a_new_home/murky -> out\secondary\diff3_a_new_home_murky_s1.pt
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
D:\LastEmbers\e1\.venv\Scripts\python.exe -u windows\train_chain.py diff a_new_home murky 100000 D:\LastEmbers\e1\out\secondary\diff3_a_new_home_murky_s1.pt --job s18_b_train_newhome --done D:\LastEmbers\e1\out\secondary\runs\a_new_home_murky.json
exit /b %ERRORLEVEL%
