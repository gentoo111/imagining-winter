# Register the on-demand scheduled task LEQueueRunner (same shape as the PhD pilot's PilotRunner:
# no trigger, runs as the logged-in user). Start it with: schtasks /run /tn LEQueueRunner
$ErrorActionPreference = 'Stop'
$E1 = 'D:\LastEmbers\e1'
New-Item -ItemType Directory -Force "$E1\queue" | Out-Null
Copy-Item "$E1\ml\forecast\windows\runner.ps1" "$E1\queue\runner.ps1" -Force
$a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $E1\queue\runner.ps1"
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$u = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
Register-ScheduledTask -TaskName 'LEQueueRunner' -Action $a -Settings $s -Principal $u -Force | Out-Null
schtasks /query /tn LEQueueRunner /fo list
