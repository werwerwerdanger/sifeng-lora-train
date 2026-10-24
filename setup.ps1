# One-shot setup for the sifeng LoRA training bundle (Windows).
# Usage: .\setup.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root "venv\Scripts\python.exe"
$mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

Write-Host "=== [1/5] Create venv (Python 3.12) ==="
if (-not (Test-Path $py)) {
    & py -3.12 -m venv (Join-Path $root "venv")
    if (-not (Test-Path $py)) { throw "Python 3.12 not found. Install it or adjust the py launcher argument." }
}
& $py -m pip install -q --upgrade pip

Write-Host "=== [2/5] Install dependencies (this takes ~10 min) ==="
& $py -m pip install -q -i $mirror numpy diffusers "transformers==4.50.0" "accelerate==1.6.0" `
    huggingface_hub loguru tqdm click librosa soundfile datasets pypinyin "py3langid==0.3.0" `
    num2words peft cutlet "fugashi[unidic-lite]" hangul-romanize spacy matplotlib modelscope `
    "pytorch_lightning==2.5.1" tensorboardX nnAudio
& $py -m pip install -q torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "=== [3/5] Download models (12.3 GB from ModelScope) ==="
$ace = Join-Path $root "models\ace"
& $py -c "from modelscope import snapshot_download as s; s('ACE-Step/ACE-Step-v1-3.5B', local_dir=r'$ace')"
& $py -c "from modelscope import snapshot_download as s; s('m-a-p/MERT-v1-330M', local_dir=r'$ace\mert')"
& $py -c "from modelscope import snapshot_download as s; s('utter-project/mHuBERT-147', local_dir=r'$ace\mhubert')"

Write-Host "=== [4/5] Rebuild dataset ==="
$repo = Join-Path $root "repo"
Push-Location $repo
& $py convert2hf_dataset.py --data_dir (Join-Path $root "train\data") --repeat_count 10 --output_name (Join-Path $root "train\ds")
Pop-Location

Write-Host "=== [5/5] Patch model paths in trainer.py ==="
$trainer = Join-Path $repo "trainer.py"
$text = Get-Content $trainer -Raw
$text = $text.Replace('D:\AI\acestep\models_ms', $ace)
Set-Content -Path $trainer -Value $text -Encoding utf8 -NoNewline

Write-Host ""
Write-Host "Setup done. Verify CUDA:"
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
Write-Host ""
Write-Host "Next:  .\train_windows.ps1 150     (150 steps, ~1 hour, quick check)"
Write-Host "Then:  .\train_windows.ps1 600     (full run, ~3-7 hours)"
