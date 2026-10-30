#requires -Version 5.1
# SPDX-License-Identifier: Apache-2.0
param(
    [ValidateSet('Q8_0','BF16','Transformers','Runtime')][string]$Variant='Q8_0',
    [string]$ModelRoot=$PSScriptRoot,
    [string]$RuntimeRoot=(Join-Path $env:LOCALAPPDATA 'Treyrin/runtime/v5-native-r3'),
    [string]$ReleaseBase='https://github.com/AlLia963/Treyrin/releases/download/v5',
    [string]$CacheDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/downloads/v5')
)
$ErrorActionPreference='Stop'
if($Variant -eq 'Runtime'){
    & (Join-Path $PSScriptRoot 'runtime/Install.ps1') -ModelRoot $ModelRoot -RuntimeRoot $RuntimeRoot -ReleaseBase $ReleaseBase -CacheDirectory $CacheDirectory
}else{
    . (Join-Path $PSScriptRoot 'runtime/Get-Artifact.ps1')
    Get-TreyrinArtifact -Variant $Variant -ModelRoot $ModelRoot -RuntimeRoot $RuntimeRoot -ReleaseBase $ReleaseBase -CacheDirectory $CacheDirectory
}
