#requires -Version 5.1
param(
    [ValidateSet('Q8_0','BF16')][string]$Variant='Q8_0',
    [string]$RuntimeRoot=(Join-Path $env:LOCALAPPDATA 'Treyrin/runtime/v5-native-r2'),
    [string]$StateDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/state/v5'),
    [int]$Port=18888,[int]$Threads=6,
    [string]$ReleaseBase='https://github.com/AlLia963/Treyrin/releases/download/v5'
)
$ErrorActionPreference='Stop'
$model=Join-Path $PSScriptRoot "$Variant/Treyrin-v5-$Variant.gguf"
if (!(Test-Path -LiteralPath $model)) {& (Join-Path $PSScriptRoot 'Download.ps1') -Variant $Variant -ReleaseBase $ReleaseBase}
if (!(Test-Path -LiteralPath (Join-Path $RuntimeRoot 'runtime-version.json'))) {& (Join-Path $PSScriptRoot 'Download.ps1') -Variant Runtime -RuntimeRoot $RuntimeRoot -ReleaseBase $ReleaseBase}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {throw "Port $Port is already in use"}
$engine=Join-Path $RuntimeRoot 'llama/llama-server.exe'
$python=Join-Path $RuntimeRoot 'python'
foreach($required in @($engine,(Join-Path $python 'python313.dll'))) {if(!(Test-Path -LiteralPath $required)){throw "Runtime component missing: $required"}}
New-Item -ItemType Directory -Path $StateDirectory -Force | Out-Null
$env:TREYRIN_STATE_DIR=[IO.Path]::GetFullPath($StateDirectory)
$env:TREYRIN_RUNTIME_DIR=Join-Path $StateDirectory 'model-code'
$env:TREYRIN_PYTHON_HOME=$python
$env:TREYRIN_LOCAL_ACCESS='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$arguments=@('-m',$model,'--alias','Treyrin-v5','--host','127.0.0.1','--port',"$Port",'-ngl','auto','-c','262144','--fit-ctx','262144','-b','128','-ub','128','-np','1','-t',"$Threads",'-n','-1','--cache-type-k','q4_0','--cache-type-v','q4_0','--flash-attn','on','--fit','on','--no-warmup')
Write-Output "Treyrin ${Variant}: http://127.0.0.1:$Port ; state: $StateDirectory"
Push-Location -LiteralPath (Split-Path -Parent $engine)
try {& $engine @arguments; if($LASTEXITCODE -ne 0){throw "llama-server exited: $LASTEXITCODE"}}finally{Pop-Location}
