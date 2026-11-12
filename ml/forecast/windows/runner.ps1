# Serial job queue for Last Embers e1 on Windows (copy of the PhD pilot's queue_runner.ps1 convention).
# Deployed as D:\LastEmbers\e1\queue\runner.ps1. Jobs are .cmd files in queue\pending, run in name
# order at below-normal priority, one at a time; output goes to queue\logs\<job>.log. A failed job
# stops the queue. Started on demand by the scheduled task "LEQueueRunner" (schtasks /run /tn LEQueueRunner).
# PhD first: before each job, wait while the PhD queue (D:\PhD\pilot\queue\running or pending) has work
# or an nh-run process exists. Each of our jobs is capped at 3 h (killed -> failed\, TIMEOUT); jobs
# checkpoint, so a killed job can be resumed.
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
