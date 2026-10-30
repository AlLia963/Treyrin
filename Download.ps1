#requires -Version 5.1
param(
    [ValidateSet('Q8_0','BF16','Transformers','Runtime')][string]$Variant='Q8_0',
    [string]$ModelRoot=$PSScriptRoot,
    [string]$RuntimeRoot=(Join-Path $env:LOCALAPPDATA 'Treyrin/runtime/v5-native-r2'),
    [string]$ReleaseBase='https://github.com/AlLia963/Treyrin/releases/download/v5'
)
$ErrorActionPreference='Stop'
[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
$catalog=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'artifacts.json') -Raw | ConvertFrom-Json
$entry=$catalog.artifacts.$Variant
if (!$entry) {throw "Unknown artifact: $Variant"}
if ($Variant -eq 'Runtime') {$target=Join-Path (Split-Path -Parent $RuntimeRoot) $entry.file}
else {$target=Join-Path $ModelRoot $entry.path}
New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
if (Test-Path -LiteralPath $target) {
    $item=Get-Item -LiteralPath $target
    if ($item.Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $entry.sha256) {throw "Existing artifact differs: $target"}
} else {
    $partial=$target+'.partial'
    $output=[IO.File]::Open($partial,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {
        $buffer=New-Object byte[] (8MB)
        for ($part=1;$part -le $entry.parts;$part++) {
            $asset=if($entry.parts -eq 1){$entry.file}else{('{0}.part{1:D3}' -f $entry.file,$part)}
            $piece=$partial+'.download'
            $uri=$ReleaseBase.TrimEnd('/')+'/'+$asset
            Write-Output "Downloading $asset"
            if ($ReleaseBase -match '^https?://') {Invoke-WebRequest -Uri $uri -OutFile $piece -UseBasicParsing}
            else {Copy-Item -LiteralPath (Join-Path $ReleaseBase $asset) -Destination $piece}
            $source=[IO.File]::OpenRead($piece)
            try {while(($count=$source.Read($buffer,0,$buffer.Length)) -gt 0){$output.Write($buffer,0,$count)}}finally{$source.Dispose()}
            Remove-Item -LiteralPath $piece
        }
    }finally{$output.Dispose()}
    if ((Get-Item -LiteralPath $partial).Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $entry.sha256) {throw 'Artifact verification failed'}
    Move-Item -LiteralPath $partial -Destination $target
}
if ($Variant -eq 'Runtime') {
    if (!(Test-Path -LiteralPath (Join-Path $RuntimeRoot 'runtime-version.json'))) {
        New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
        Expand-Archive -LiteralPath $target -DestinationPath $RuntimeRoot -Force
    }
    $version=Get-Content -LiteralPath (Join-Path $RuntimeRoot 'runtime-version.json') -Raw | ConvertFrom-Json
    if ($version.id -ne $entry.runtime_id) {throw 'Installed runtime has a different version'}
}
Write-Output "Verified: $target"
