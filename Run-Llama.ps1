#requires -Version 5.1
# SPDX-License-Identifier: Apache-2.0
param(
    [ValidateSet('Q8_0','BF16')][string]$Variant='Q8_0',
    [string]$RuntimeRoot=(Join-Path $env:LOCALAPPDATA 'Treyrin/runtime/v5-native-r3'),
    [string]$StateDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/state/v5'),
    [string]$ReleaseBase='https://github.com/AlLia963/Treyrin/releases/download/v5',
    [string]$CacheDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/downloads/v5'),
    [int]$Port=18888,[int]$Threads=6,[int]$ContextSize=262144,
    [string]$GpuLayers='auto',[ValidateSet('on','off')][string]$Fit='on',
    [string]$BindHost='127.0.0.1',[string]$ApiKey
)
& (Join-Path $PSScriptRoot 'runtime/Start.ps1') -ModelRoot $PSScriptRoot @PSBoundParameters
