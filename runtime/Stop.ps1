#requires -Version 5.1
# SPDX-License-Identifier: Apache-2.0
param([string]$StateDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/state/v5'))
$ErrorActionPreference='Stop'
$record=Get-Content -Raw -LiteralPath (Join-Path $StateDirectory 'server.json') | ConvertFrom-Json
$process=Get-CimInstance Win32_Process -Filter "ProcessId=$($record.pid)"
if(!$process){Write-Output 'Server is already stopped';return}
if([IO.Path]::GetFullPath($process.ExecutablePath) -ne [IO.Path]::GetFullPath($record.executable) -or
   !$process.CommandLine -or $process.CommandLine.IndexOf($record.model,[StringComparison]::OrdinalIgnoreCase) -lt 0){throw 'PID belongs to a different process; nothing was stopped'}
Stop-Process -Id $record.pid -ErrorAction Stop
Write-Output "Stopped Treyrin PID $($record.pid)"
