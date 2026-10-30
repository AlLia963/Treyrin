#requires -Version 5.1
# SPDX-License-Identifier: Apache-2.0
param(
    [ValidateSet('Q8_0','BF16')][string]$Variant='Q8_0',
    [string]$ModelRoot=(Split-Path -Parent $PSScriptRoot),
    [string]$RuntimeRoot=(Join-Path $env:LOCALAPPDATA 'Treyrin/runtime/v5-native-r3'),
    [string]$StateDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/state/v5'),
    [string]$ReleaseBase='https://github.com/AlLia963/Treyrin/releases/download/v5',
    [string]$CacheDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/downloads/v5'),
    [int]$Port=18888,[int]$Threads=6,[int]$ContextSize=262144,
    [string]$GpuLayers='auto',[ValidateSet('on','off')][string]$Fit='on',
    [string]$BindHost='127.0.0.1',[string]$ApiKey
)
$ErrorActionPreference='Stop'
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw "Port $Port is already in use"}
. (Join-Path $PSScriptRoot 'Get-Artifact.ps1')
$model=Get-TreyrinArtifact -Variant $Variant -ModelRoot $ModelRoot -RuntimeRoot $RuntimeRoot -ReleaseBase $ReleaseBase -CacheDirectory $CacheDirectory
$installed=& (Join-Path $PSScriptRoot 'Install.ps1') -ModelRoot $ModelRoot -RuntimeRoot $RuntimeRoot -ReleaseBase $ReleaseBase -CacheDirectory $CacheDirectory
$engine=Join-Path $installed 'llama/llama-server.exe'
$state=[IO.Path]::GetFullPath($StateDirectory)
$logs=Join-Path $state 'logs'
if(Test-Path -LiteralPath (Join-Path $state 'server.json')){
    $previous=Get-Content -Raw -LiteralPath (Join-Path $state 'server.json') | ConvertFrom-Json
    $active=Get-CimInstance Win32_Process -Filter "ProcessId=$($previous.pid)"
    if($active -and $active.CommandLine -and $active.CommandLine.IndexOf($state,[StringComparison]::OrdinalIgnoreCase) -ge 0){throw 'This state directory is already used by a running server'}
    if($active -and $active.ExecutablePath -eq $previous.executable){throw 'A server recorded for this state directory is still running'}
}
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$configuration=@{
    TREYRIN_STATE_DIR=$state
    TREYRIN_RUNTIME_DIR=(Join-Path $state 'model-code')
    TREYRIN_PYTHON_HOME=(Join-Path $installed 'python')
    TREYRIN_LOCAL_ACCESS='1'
    PYTHONNOUSERSITE='1'
    PYTHONDONTWRITEBYTECODE='1'
    PATH=((Join-Path $installed 'llama')+';'+$env:SystemRoot+'\System32;'+$env:SystemRoot)
}
$arguments=@('-m',$model,'--alias','Treyrin-v5','--host',$BindHost,'--port',"$Port",
    '-ngl',$GpuLayers,'-c',"$ContextSize",'--fit-ctx',"$ContextSize",'-b','128','-ub','128',
    '-np','1','-t',"$Threads",'-n','-1','--cache-type-k','q4_0','--cache-type-v','q4_0',
    '--flash-attn','on','--fit',$Fit,'--no-warmup')
if($ApiKey){$arguments+=@('--api-key',$ApiKey)}
# Start-Process joins ArgumentList; quote values so clone/state paths may contain spaces.
$quotedArguments=@($arguments | ForEach-Object {'"'+($_ -replace '(\\*)"','$1$1\"' -replace '(\\+)$','$1$1')+'"'})
$saved=@{}
try{
    foreach($entry in $configuration.GetEnumerator()){
        $saved[$entry.Key]=[Environment]::GetEnvironmentVariable($entry.Key,'Process')
        [Environment]::SetEnvironmentVariable($entry.Key,$entry.Value,'Process')
    }
    $process=Start-Process -FilePath $engine -ArgumentList $quotedArguments -WorkingDirectory (Split-Path -Parent $engine) `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'server.stdout.log') `
        -RedirectStandardError (Join-Path $logs 'server.stderr.log')
}finally{foreach($name in $saved.Keys){[Environment]::SetEnvironmentVariable($name,$saved[$name],'Process')}}
@{pid=$process.Id;executable=$engine;model=$model;port=$Port;started_at=(Get-Date).ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $state 'server.json') -Encoding UTF8
$ready=$false
for($attempt=0;$attempt -lt 120;$attempt++){
    $process.Refresh()
    if($process.HasExited){throw "Server exited ($($process.ExitCode)); inspect $logs\server.stderr.log"}
    try{$health=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2;if($health.status -eq 'ok'){$ready=$true;break}}catch{}
    Start-Sleep -Milliseconds 500
}
if(!$ready){throw "Server is still starting (PID $($process.Id)); inspect $logs\server.stderr.log"}
Write-Host "Treyrin $Variant ready: http://127.0.0.1:$Port"
[pscustomobject]@{Pid=$process.Id;Url="http://127.0.0.1:$Port";Model=$model;Runtime=$installed;State=$state;Logs=$logs}
