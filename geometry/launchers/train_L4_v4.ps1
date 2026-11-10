param(
    [string]$Python = "C:\ProgramData\Anaconda3\envs\mld\python.exe",
    [string]$Workspace = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Workspace) {
    $Workspace = Split-Path -Parent $Repo
}
$Source = Join-Path $Workspace "data\ct\teachers\v4\L4"
$PreviousLayer = Join-Path $Workspace "output\L3_v3\layer_save_30000.pth"
if (-not $Output) {
    $Output = Join-Path $Workspace "output\L4_v4"
}

foreach ($RequiredPath in @($Python, $Repo, $Source, $PreviousLayer)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path does not exist: $RequiredPath"
    }
}

if (Test-Path -LiteralPath $Output) {
    throw "Output directory already exists: $Output"
}

Set-Location -LiteralPath $Repo

& $Python -m geometry.train `
    -s $Source `
    -m $Output `
    --load_layer $PreviousLayer `
    --eval `
    --save_layer `
    -a `
    --random_background

if ($LASTEXITCODE -ne 0) {
    throw "L4_v4 training failed with exit code $LASTEXITCODE"
}
