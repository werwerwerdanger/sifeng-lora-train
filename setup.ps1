# One-shot setup for the sifeng LoRA training bundle (Windows).
# Usage: .\setup.ps1
$ErrorActionPreference = "Stop"
$root   = $PSScriptRoot
$py     = Join-Path $root "venv\Scripts\python.exe"
$mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

function Step($n, $text) { Write-Host ""; Write-Host "=== [$n] $text ===" -ForegroundColor Cyan }

Step "1/6" "Create venv (Python 3.12)"
if (-not (Test-Path $py)) {
    $made = $false
    foreach ($launcher in @("py -3.12", "py -3", "python")) {
        try {
            $parts = $launcher.Split(" ")
            if ($parts.Length -ge 2) {
                & $parts[0] $parts[1] -m venv (Join-Path $root "venv")
            } else {
                & $parts[0] -m venv (Join-Path $root "venv")
            }
        } catch { }
        if (Test-Path $py) { Write-Host "venv created via: $launcher"; $made = $true; break }
    }
    if (-not $made) { throw "No usable Python found. Install Python 3.12 x64 and re-run." }
}
& $py -m pip install --upgrade pip -i $mirror --progress-bar on
if ($LASTEXITCODE -ne 0) { Write-Host "pip upgrade skipped (network), continuing anyway" -ForegroundColor Yellow }

Step "2/6" "Install Python dependencies (~10 min)"
& $py -m pip install -i $mirror --progress-bar on numpy diffusers "transformers==4.50.0" "accelerate==1.6.0" `
    huggingface_hub loguru tqdm click librosa soundfile datasets pypinyin "py3langid==0.3.0" `
    num2words peft cutlet "fugashi[unidic-lite]" hangul-romanize spacy matplotlib modelscope `
    "pytorch_lightning==2.5.1" tensorboardX nnAudio

Step "3/6" "Install torch cu128 (~2.5 GB, tries official then mirror)"
$torchOk = $false
try {
    & $py -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --progress-bar on
    $torchOk = $true
} catch {
    Write-Host "Official index failed, trying aliyun mirror..." -ForegroundColor Yellow
}
if (-not $torchOk) {
    & $py -m pip install torch torchaudio torchvision --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128 --progress-bar on
}
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"

Step "4/6" "Download models (12.3 GB from ModelScope)"
$ace = Join-Path $root "models\ace"
& $py -c "from modelscope import snapshot_download as s; s('ACE-Step/ACE-Step-v1-3.5B', local_dir=r'$ace')"
& $py -c "from modelscope import snapshot_download as s; s('m-a-p/MERT-v1-330M', local_dir=r'$ace\mert')"
& $py -c "from modelscope import snapshot_download as s; s('utter-project/mHuBERT-147', local_dir=r'$ace\mhubert')"

Step "5/6" "Rebuild dataset"
$repo = Join-Path $root "repo"
Push-Location $repo
& $py convert2hf_dataset.py --data_dir (Join-Path $root "train\data") --repeat_count 10 --output_name (Join-Path $root "train\ds")
Pop-Location
$dsCount = (Get-ChildItem (Join-Path $root "train\ds") -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "dataset files: $dsCount"

Step "6/6" "Patch model paths in trainer.py"
$trainer = Join-Path $repo "trainer.py"
$src = Get-Content $trainer -Raw
if ($src -match 'ACESTEP_MODELS') {
    Write-Host "trainer.py already reads ACESTEP_MODELS (no rewrite needed)"
} else {
    Write-Host "WARNING: trainer.py does not support ACESTEP_MODELS, falling back to text patch" -ForegroundColor Yellow
    $text = $src.Replace('D:\AI\acestep\models_ms', $ace)
    Set-Content -Path $trainer -Value $text -Encoding utf8 -NoNewline
}
$env:ACESTEP_MODELS = $ace
Write-Host "ACESTEP_MODELS = $ace"
& $py -c "import ast,io; ast.parse(io.open(r'$trainer',encoding='utf-8').read()); print('trainer.py syntax OK')"

Write-Host ""
Write-Host "=== Setup finished ===" -ForegroundColor Green
Write-Host "Next:  .\train_windows.ps1 150     (quick check, ~1 hour)"
Write-Host "Then:  .\train_windows.ps1 600     (full run)"
