# Second CPU-only lane for SMALL jobs (driver 2026-11-06): D:\LastEmbers\e1\queue-cpu2\runner.ps1, scheduled task
# "LEQueueRunnerCPU2" (register_task_cpu.ps1 -Lane 2). Same semantics as runner_cpu.ps1 / runner.ps1 (name order,
# serial within the lane, <= 3 h per job, a failed job stops the lane, PhD first) and runs concurrently with
# queue-cpu, which is reserved for the S1-11b PPO segments (p1_ppo3b, ~12 of the 20 logical threads). Thread cap: the
# lane exports OMP/MKL/TORCH/RAYON thread counts of 6 and CUDA_VISIBLE_DEVICES=-1 to every job; jobs may lower them but
# must not raise them (6 threads total for this lane).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
foreach ($d in 'pending', 'running', 'done', 'failed', 'hold', 'cancelled', 'logs') { New-Item -ItemType Directory -Force $d | Out-Null }
$lock = 'runner.pid'
if ((Test-Path $lock) -and (Get-Process -Id (Get-Content $lock) -ErrorAction SilentlyContinue)) { exit 0 }
Set-Content $lock $PID
$MaxMinutes = 180
function Log($msg) { Add-Content 'status.log' ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg) }
function PhdBusy {
  $r = @(Get-ChildItem 'D:\PhD\pilot\queue\running' -File -ErrorAction SilentlyContinue)
  $p = @(Get-ChildItem 'D:\PhD\pilot\queue\pending' -File -Filter *.cmd -ErrorAction SilentlyContinue)
  $n = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'nh-run' })
  if ($r.Count -or $p.Count -or $n.Count) { return "running=$($r.Count) pending=$($p.Count) nh-run=$($n.Count)" }
  return $null
}
try {
  while ($true) {
    $job = Get-ChildItem 'pending' -File -Filter *.cmd | Sort-Object Name | Select-Object -First 1
    if (-not $job) { Log 'IDLE queue empty'; break }
    $busy = PhdBusy
    if ($busy) {
      Log "WAIT phd $busy (next: $($job.Name))"
      while (PhdBusy) { Start-Sleep 60 }
      Log 'RESUME phd idle'
      continue
    }
    Move-Item $job.FullName "running\$($job.Name)"
    foreach ($v in 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TORCH_THREADS', 'RAYON_NUM_THREADS') { Set-Item "env:$v" '6' }
    $env:CUDA_VISIBLE_DEVICES = '-1'; $env:DEVICE = 'cpu'
    Log "START $($job.Name)"
    $start = Get-Date
    $p = Start-Process cmd.exe -ArgumentList '/c', "running\$($job.Name) > logs\$($job.BaseName).log 2>&1" -NoNewWindow -PassThru
    $null = $p.Handle  # Windows PowerShell 5.1 loses ExitCode unless the handle is opened early
    $p.PriorityClass = 'BelowNormal'
    if (-not $p.WaitForExit($MaxMinutes * 60 * 1000)) {
      & taskkill.exe /T /F /PID $p.Id | Out-Null
      $p.WaitForExit()
      Move-Item "running\$($job.Name)" 'failed\'
      Log "TIMEOUT $($job.Name) killed after ${MaxMinutes}min"
      break
    }
    $mins = [int]((Get-Date) - $start).TotalMinutes
    if ($p.ExitCode -eq 0) { Move-Item "running\$($job.Name)" 'done\' -Force; Log "DONE $($job.Name) ${mins}min" }
    else { Move-Item "running\$($job.Name)" 'failed\' -Force; Log "FAILED $($job.Name) rc=$($p.ExitCode) ${mins}min"; break }
  }
} finally { Remove-Item $lock -ErrorAction SilentlyContinue }
