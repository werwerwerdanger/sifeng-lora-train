param([int]$Steps = 100, [int]$Workers = 8)
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$py   = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$repo = Join-Path $root "repo"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:ACESTEP_MODELS = Join-Path $root "models\ace"
Set-Location $repo

function Run-Train([int]$nw) {
    & $py trainer.py `
        --dataset_path (Join-Path $root "train/ds") `
        --checkpoint_dir (Join-Path $root "models/ace") `
        --exp_name sifeng_lora `
        --precision bf16-mixed `
        --max_steps $Steps `
        --every_n_train_steps 25 `
        --accumulate_grad_batches 2 `
        --lora_config_path (Join-Path $root "train/sifeng_lora.json") `
        --logger_dir (Join-Path $root "train/logs") `
        --num_workers $nw
    return $LASTEXITCODE
}

Write-Host "Training $Steps steps with $Workers dataloader workers (i7-13700: use 8)"
$code = Run-Train $Workers
if ($code -ne 0) {
    Write-Host "Worker mode failed (exit $code), falling back to single-process loader..."
    $code = Run-Train 0
}
if ($code -eq 0) {
    Write-Host "Finished. Checkpoints: $root\train\logs"
} else {
    Write-Host "Training failed (exit $code). Check the console output above."
}
