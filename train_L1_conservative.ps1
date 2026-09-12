$ErrorActionPreference = "Stop"

$Python = "C:\ProgramData\Anaconda3\envs\mld\python.exe"
$Repo = "E:\3dguass\Multi-Layer-Anatomy-GS-Training"
$Output = "E:\3dguass\output\L1_conservative"

if (Test-Path -LiteralPath $Output) {
    throw "Output directory already exists: $Output. Rename it or choose a new output path to avoid overwriting a previous run."
}

Set-Location -LiteralPath $Repo

& $Python train.py `
    -s E:\3dguass\teachers\L1 `
    -m $Output `
    --load_layer E:\3dguass\output\L0\layer_save_30000.pth `
    --eval `
    --save_layer `
    -a `
    --random_background `
    --inactive_purge_threshold -1 `
    --prune_from_iter 3000 `
    --min_active_points 15000 `
    --max_prune_fraction 0.10 `
    --min_opacity 0.0005 `
    --densify_grad_threshold 0.00005 `
    --densify_until_iter 20000
