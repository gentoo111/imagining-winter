# Register the on-demand scheduled task LEQueueRunnerCPU for the CPU-only lane (S1-11a; same shape as
# register_task.ps1 / LEQueueRunner: no trigger, runs as the logged-in user). Start: schtasks /run /tn LEQueueRunnerCPU
param([int]$Lane = 1)  # -Lane 2: the small-jobs lane queue-cpu2 / LEQueueRunnerCPU2 (runner_cpu2.ps1)
$ErrorActionPreference = 'Stop'
$E1 = 'D:\LastEmbers\e1'
$Q = if ($Lane -eq 2) { 'queue-cpu2' } else { 'queue-cpu' }
$T = if ($Lane -eq 2) { 'LEQueueRunnerCPU2' } else { 'LEQueueRunnerCPU' }
$R = if ($Lane -eq 2) { 'runner_cpu2.ps1' } else { 'runner_cpu.ps1' }
New-Item -ItemType Directory -Force "$E1\$Q" | Out-Null
Copy-Item "$E1\ml\forecast\windows\$R" "$E1\$Q\runner.ps1" -Force
$a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $E1\$Q\runner.ps1"
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$u = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
Register-ScheduledTask -TaskName $T -Action $a -Settings $s -Principal $u -Force | Out-Null
schtasks /query /tn $T /fo list
