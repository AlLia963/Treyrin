#requires -Version 5.1
# SPDX-License-Identifier: Apache-2.0
param(
    [string]$ModelRoot=(Split-Path -Parent $PSScriptRoot),
    [string]$RuntimeRoot=(Join-Path $env:LOCALAPPDATA 'Treyrin/runtime/v5-native-r3'),
    [string]$ReleaseBase='https://github.com/AlLia963/Treyrin/releases/download/v5',
    [string]$CacheDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/downloads/v5')
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'Get-Artifact.ps1')
$catalog=Get-Content -Raw -LiteralPath (Join-Path $ModelRoot 'artifacts.json') | ConvertFrom-Json
$expected=$catalog.artifacts.Runtime
$marker=Join-Path $RuntimeRoot 'runtime-version.json'
if(Test-Path -LiteralPath $marker){
    $version=Get-Content -Raw -LiteralPath $marker | ConvertFrom-Json
    if($version.id -ne $expected.runtime_id){throw "Different runtime installed; choose a new RuntimeRoot: $RuntimeRoot"}
    $hashes=Get-Content -Raw -LiteralPath (Join-Path $RuntimeRoot 'files.json') | ConvertFrom-Json
    foreach($name in @('llama/llama-server.exe','llama/libllama.dll','llama/libllama-server-impl.dll',
        'llama/ggml-cuda.dll','llama/onnxruntime.dll','python/python313.dll',
        'python/Lib/site-packages/torch/lib/torch_cpu.dll')){
        $file=Join-Path $RuntimeRoot $name
        if(!(Test-Path -LiteralPath $file) -or (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $hashes.$name){throw "Installed runtime differs; preserved: $file"}
    }
    return $RuntimeRoot
}
if(Test-Path -LiteralPath $RuntimeRoot){throw "RuntimeRoot already exists without a valid installation; choose an empty location: $RuntimeRoot"}
$archive=Get-TreyrinArtifact -Variant Runtime -ModelRoot $ModelRoot -RuntimeRoot $RuntimeRoot -ReleaseBase $ReleaseBase -CacheDirectory $CacheDirectory
$absolute=[IO.Path]::GetFullPath($RuntimeRoot)
$staging=$absolute+'.install-'+[Guid]::NewGuid().ToString('N')
New-Item -ItemType Directory -Path (Split-Path -Parent $absolute) -Force | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $staging
$version=Get-Content -Raw -LiteralPath (Join-Path $staging 'runtime-version.json') | ConvertFrom-Json
if($version.id -ne $expected.runtime_id){throw 'Runtime archive version differs'}
$hashes=Get-Content -Raw -LiteralPath (Join-Path $staging 'files.json') | ConvertFrom-Json
foreach($property in $hashes.PSObject.Properties){
    $file=[IO.Path]::GetFullPath((Join-Path $staging $property.Name))
    if(!$file.StartsWith($staging+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Invalid runtime manifest path'}
    if(!(Test-Path -LiteralPath $file) -or (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $property.Value){throw "Runtime verification failed: $($property.Name)"}
}
Move-Item -LiteralPath $staging -Destination $absolute
Write-Host "Installed verified runtime: $absolute"
return $absolute
