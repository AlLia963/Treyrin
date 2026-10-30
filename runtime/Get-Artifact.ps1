# SPDX-License-Identifier: Apache-2.0
function Get-TreyrinArtifact {
    param([string]$Variant,[string]$ModelRoot,[string]$RuntimeRoot,[string]$ReleaseBase,
          [string]$CacheDirectory=(Join-Path $env:LOCALAPPDATA 'Treyrin/downloads/v5'))
    $ErrorActionPreference='Stop'
    $catalog=Get-Content -Raw -LiteralPath (Join-Path $ModelRoot 'artifacts.json') | ConvertFrom-Json
    $entry=$catalog.artifacts.$Variant
    if(!$entry){throw "Unknown artifact: $Variant"}
    $target=if($Variant -eq 'Runtime'){Join-Path $CacheDirectory $entry.file}else{Join-Path $ModelRoot $entry.path}
    New-Item -ItemType Directory -Path (Split-Path -Parent $target),$CacheDirectory -Force | Out-Null
    if(Test-Path -LiteralPath $target){
        if((Get-Item -LiteralPath $target).Length -ne [long]$entry.bytes -or
           (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $entry.sha256){throw "Existing file differs; preserved: $target"}
        return $target
    }
    $pieces=@()
    for($part=1;$part -le [int]$entry.parts;$part++){
        $name=if($entry.parts -eq 1){$entry.file}else{('{0}.part{1:D3}' -f $entry.file,$part)}
        $piece=Join-Path $CacheDirectory $name
        $expected=[Math]::Min([long]$catalog.part_bytes,[long]$entry.bytes-([long]($part-1)*[long]$catalog.part_bytes))
        if($entry.parts -eq 1){$expected=[long]$entry.bytes}
        if(!(Test-Path -LiteralPath $piece) -or (Get-Item -LiteralPath $piece).Length -ne $expected){
            Write-Host "Downloading $name"
            $partial=$piece+'.download'
            if($ReleaseBase -match '^https?://'){
                $uri=$ReleaseBase.TrimEnd('/')+'/'+$name
                & curl.exe --fail --location --retry 3 --retry-delay 3 --connect-timeout 20 --continue-at - --output $partial $uri
                if($LASTEXITCODE -ne 0){throw "Download failed; partial data is preserved: $partial"}
            }else{Copy-Item -LiteralPath (Join-Path $ReleaseBase $name) -Destination $partial}
            if((Get-Item -LiteralPath $partial).Length -ne $expected){throw "Unexpected size: $partial"}
            Move-Item -LiteralPath $partial -Destination $piece -Force
        }
        $pieces+=,$piece
    }
    if($entry.parts -eq 1){
        if((Get-FileHash -LiteralPath $pieces[0] -Algorithm SHA256).Hash -ne $entry.sha256){throw 'Artifact SHA256 differs'}
        return $pieces[0]
    }
    $temporary=$target+'.partial'
    $output=[IO.File]::Open($temporary,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try{foreach($piece in $pieces){$input=[IO.File]::OpenRead($piece);try{$input.CopyTo($output,8MB)}finally{$input.Dispose()}}}finally{$output.Dispose()}
    if((Get-Item -LiteralPath $temporary).Length -ne [long]$entry.bytes -or
       (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash -ne $entry.sha256){throw "SHA256 verification failed; files preserved: $temporary"}
    Move-Item -LiteralPath $temporary -Destination $target
    return $target
}
