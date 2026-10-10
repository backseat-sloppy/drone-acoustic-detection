<#
.SYNOPSIS
    Download the public audio datasets into data/, untouched and checksum-verified.

.DESCRIPTION
    Zenodo caps each connection at roughly 50-100 KB/s, so every file is split
    into byte ranges fetched over many connections at once. Ranges land in a
    <file>.parts folder, so a stopped run resumes where it left off. A file only
    gets its final name after its MD5 matches the one the host publishes.

    Nothing is unzipped or modified. The host's own record is saved next to each
    dataset as _source_record.json.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
    powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1 -Only ddl,halmstad
#>
param(
    # dronenoise, halmstad, ddl, fsd50k
    [string[]]$Only = @('dronenoise', 'halmstad', 'ddl', 'fsd50k'),
    [int]$Connections = 16,
    [int]$ChunkMB = 32,
    [string]$DataRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'data')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# powershell -File hands "-Only a,b" over as one string, so split it here.
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
$ChunkBytes = [int64]$ChunkMB * 1MB
$MaxRounds = 8

# Only the real recordings from DDL; the synthesized zip is deliberately skipped.
$Datasets = [ordered]@{
    dronenoise = @{ Folder = 'DroneNoise_Salford'; Host = 'figshare'; Id = 22133411; Keep = $null }
    halmstad   = @{ Folder = 'Halmstad'; Host = 'zenodo'; Id = 5500576; Keep = $null }
    ddl        = @{ Folder = 'DDL_Surrey'; Host = 'zenodo'; Id = 6459183; Keep = @('MLSP_2022_Real_Data.zip') }
    fsd50k     = @{ Folder = 'FSD50K'; Host = 'zenodo'; Id = 4060432; Keep = $null }
}

function Get-FileList($ds, $dir) {
    $recordPath = Join-Path $dir '_source_record.json'
    if ($ds.Host -eq 'zenodo') {
        $api = "https://zenodo.org/api/records/$($ds.Id)"
        & curl.exe -sS -L --fail --retry 5 -o $recordPath $api
        if ($LASTEXITCODE -ne 0) { throw "Could not fetch $api" }
        $rec = Get-Content $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $files = $rec.files | ForEach-Object {
            [pscustomobject]@{
                # Halmstad's key carries a folder prefix; keep only the file name.
                Name = Split-Path $_.key -Leaf
                Key  = $_.key
                Size = [int64]$_.size
                Md5  = ($_.checksum -replace '^md5:', '')
                Url  = $_.links.self
            }
        }
    }
    else {
        $api = "https://api.figshare.com/v2/articles/$($ds.Id)"
        & curl.exe -sS -L --fail --retry 5 -o $recordPath $api
        if ($LASTEXITCODE -ne 0) { throw "Could not fetch $api" }
        $rec = Get-Content $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $files = $rec.files | ForEach-Object {
            [pscustomobject]@{
                Name = $_.name
                Key  = $_.name
                Size = [int64]$_.size
                Md5  = $_.computed_md5
                Url  = $_.download_url
            }
        }
    }
    if ($ds.Keep) { $files = $files | Where-Object { $ds.Keep -contains $_.Key } }
    # figshare lists one DroneNoise WAV twice (same name, same MD5); keep one copy.
    $files = $files | Group-Object Name, Md5 | ForEach-Object { $_.Group[0] }
    return @($files)
}

function Get-Chunks($file, $dir) {
    $partsDir = Join-Path $dir ($file.Name + '.parts')
    $count = [int][math]::Max(1, [math]::Ceiling($file.Size / $ChunkBytes))
    for ($i = 0; $i -lt $count; $i++) {
        $start = [int64]$i * $ChunkBytes
        $end = [math]::Min($start + $ChunkBytes, $file.Size) - 1
        [pscustomobject]@{
            Path   = Join-Path $partsDir ('{0:D5}' -f $i)
            Start  = $start
            End    = $end
            Length = $end - $start + 1
            Url    = $file.Url
            Ranged = ($count -gt 1)
        }
    }
}

function Test-Chunk($chunk) {
    (Test-Path -LiteralPath $chunk.Path) -and ((Get-Item -LiteralPath $chunk.Path).Length -eq $chunk.Length)
}

function Invoke-ChunkDownload($chunks, $configPath) {
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($c in $chunks) {
        if (Test-Path -LiteralPath $c.Path) { Remove-Item -LiteralPath $c.Path -Force }
        # "next" separates transfers; a trailing one makes curl abort the whole batch.
        if ($lines.Count) { $lines.Add('next') }
        $lines.Add('url = "' + $c.Url + '"')
        $lines.Add('output = "' + ($c.Path -replace '\\', '/') + '"')
        if ($c.Ranged) { $lines.Add('range = "' + $c.Start + '-' + $c.End + '"') }
        $lines.Add('location')
        $lines.Add('fail')
        $lines.Add('retry = 8')
        $lines.Add('retry-delay = 3')
        $lines.Add('retry-all-errors')
        $lines.Add('connect-timeout = 30')
        $lines.Add('speed-limit = 1000')
        $lines.Add('speed-time = 60')
    }
    [IO.File]::WriteAllLines($configPath, $lines, (New-Object Text.UTF8Encoding($false)))
    & curl.exe --parallel --parallel-max $Connections --parallel-immediate --silent --show-error --config $configPath
}

function Join-Chunks($chunks, $target) {
    $out = [IO.File]::Open($target, [IO.FileMode]::Create, [IO.FileAccess]::Write)
    try {
        foreach ($c in $chunks) {
            $in = [IO.File]::OpenRead($c.Path)
            try { $in.CopyTo($out, 4MB) } finally { $in.Dispose() }
        }
    }
    finally { $out.Dispose() }
}

$unknown = @($Only | Where-Object { -not $Datasets.Contains($_) })
if ($unknown) { throw "Unknown dataset(s): $($unknown -join ', '). Valid: $($Datasets.Keys -join ', ')" }

$failed = @()
foreach ($key in $Datasets.Keys) {
    if ($Only -notcontains $key) { continue }
    $ds = $Datasets[$key]
    $dir = Join-Path $DataRoot $ds.Folder
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    $files = @(Get-FileList $ds $dir)
    $total = ($files | Measure-Object Size -Sum).Sum
    Write-Host ("[{0}] {1}: {2} files, {3:N2} GB" -f (Get-Date -Format HH:mm:ss), $ds.Folder, $files.Count, ($total / 1GB))

    $todo = @($files | Where-Object {
            $final = Join-Path $dir $_.Name
            -not ((Test-Path -LiteralPath $final) -and ((Get-Item -LiteralPath $final).Length -eq $_.Size))
        })
    if (-not $todo) { Write-Host "    already complete"; continue }

    $chunkMap = @{}
    foreach ($f in $todo) {
        New-Item -ItemType Directory -Force -Path (Join-Path $dir ($f.Name + '.parts')) | Out-Null
        $chunkMap[$f.Name] = @(Get-Chunks $f $dir)
    }
    $allChunks = @($chunkMap.Values | ForEach-Object { $_ })
    $configPath = Join-Path $dir '_curl_chunks.cfg'

    for ($round = 1; $round -le $MaxRounds; $round++) {
        $missing = @($allChunks | Where-Object { -not (Test-Chunk $_) })
        if (-not $missing) { break }
        $bytes = ($missing | Measure-Object Length -Sum).Sum
        Write-Host ("    round {0}: {1} of {2} pieces to fetch, {3:N2} GB" -f $round, $missing.Count, $allChunks.Count, ($bytes / 1GB))
        Invoke-ChunkDownload $missing $configPath
        if (@($allChunks | Where-Object { -not (Test-Chunk $_) }).Count -eq $missing.Count) {
            Write-Host "    no progress this round, giving up on $($ds.Folder)"
            break
        }
    }
    Remove-Item -LiteralPath $configPath -Force -ErrorAction SilentlyContinue

    foreach ($f in $todo) {
        $chunks = $chunkMap[$f.Name]
        if (@($chunks | Where-Object { -not (Test-Chunk $_) })) {
            Write-Host "    INCOMPLETE $($f.Name) (rerun to resume)"
            $failed += "$($ds.Folder)/$($f.Name): incomplete"
            continue
        }
        $tmp = Join-Path $dir ($f.Name + '.unverified')
        Join-Chunks $chunks $tmp
        $md5 = (Get-FileHash -LiteralPath $tmp -Algorithm MD5).Hash.ToLower()
        if ($md5 -ne $f.Md5.ToLower()) {
            Write-Host "    CHECKSUM MISMATCH $($f.Name): got $md5, host says $($f.Md5)"
            $failed += "$($ds.Folder)/$($f.Name): checksum mismatch"
            continue
        }
        Move-Item -LiteralPath $tmp -Destination (Join-Path $dir $f.Name) -Force
        Remove-Item -LiteralPath (Join-Path $dir ($f.Name + '.parts')) -Recurse -Force
        Write-Host ("    verified {0} ({1:N1} MB)" -f $f.Name, ($f.Size / 1MB))
    }
}

if ($failed) {
    Write-Host "`nFAILED:"
    $failed | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host ("`n[{0}] All requested datasets downloaded and checksum-verified." -f (Get-Date -Format HH:mm:ss))
