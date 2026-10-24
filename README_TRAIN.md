# 司凤风格 LoRA 训练包

在另一台带 GPU 的机器上训练 ACE-Step 的「司凤风格」LoRA。
已包含：**打过 8 处补丁**的 ACE-Step 代码、57 段 30 秒司凤伴奏训练片段（含风格标签）、LoRA 配置、启动脚本。

## 1. 获取本包

私有仓库：https://github.com/werwerwerdanger/sifeng-lora-train

```bash
# A. 浏览器登录 GitHub -> 仓库页 -> Code -> Download ZIP（最省事）
# B. 已装 gh 并登录同一账号：
gh repo clone werwerwerdanger/sifeng-lora-train
# C. git + 私有仓库 Token：
git clone https://<TOKEN>@github.com/werwerwerdanger/sifeng-lora-train.git
```

国内直连 GitHub 不通时，在 URL 前加 `https://gh-proxy.com/`。
放到 **`D:\sifeng-lora`**（下面命令都按这个路径）。

## 2. 一键安装

```powershell
cd D:\sifeng-lora
Get-ChildItem -Recurse | Unblock-File      # 若是下载的 ZIP，先解除锁定
.\setup.ps1                                # 建环境 + 装依赖 + 下模型 + 建数据集 + 改路径
```

`setup.ps1` 依次做：创建 Python 3.12 venv → 装依赖（含 torch cu128，约 10 分钟）→
从 ModelScope 下 12.3G 模型 → 重建 HF 数据集 → 把 trainer.py 里的 MERT/mHuBERT 路径改成本机路径。

## 3. 目录结构

```
D:\sifeng-lora\
  repo\                 已打补丁的 ACE-Step 代码
  train\
    data\               57 个 mp3 + _prompt.txt + _lyrics.txt
    ds\                 数据集（setup 生成）
    sifeng_lora.json    LoRA 配置 r=16 / alpha=32
  models\ace\           模型（setup 下载）
    ace_step_transformer / music_dcae_f8c8 / music_vocoder / umt5-base
    mert\ mhubert\
  venv\
```

## 4. 开始训练

```powershell
.\train_windows.ps1 150     # 先跑 150 步验证方向（约 1 小时）
.\train_windows.ps1 600     # 再跑满 600 步（约 3~7 小时）
```

参数：
- 第 1 个：训练步数
- 第 2 个：dataloader worker 数，默认 8（i7-13700 的 16C/24T 够用）；worker 模式崩溃会自动回退单进程

训练期间别开游戏/模拟器，显存占用约 7.6G（bf16-mixed），12G 卡有余量。

## 5. 产物

- checkpoint：`train\logs\lightning_logs\<时间>_sifeng_lora\checkpoints\*.ckpt`
- 推理时用 `lora_name_or_path=<ckpt 路径>`、`lora_weight=1.0`

## 6. 手动步骤（setup.ps1 失败时）

```powershell
# 环境
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  numpy diffusers "transformers==4.50.0" "accelerate==1.6.0" huggingface_hub loguru tqdm click `
  librosa soundfile datasets pypinyin "py3langid==0.3.0" num2words peft cutlet "fugashi[unidic-lite]" `
  hangul-romanize spacy matplotlib modelscope "pytorch_lightning==2.5.1" tensorboardX nnAudio
.\venv\Scripts\python.exe -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# 模型
.\venv\Scripts\python.exe -c "from modelscope import snapshot_download as s; s('ACE-Step/ACE-Step-v1-3.5B', local_dir=r'D:\sifeng-lora\models\ace')"
.\venv\Scripts\python.exe -c "from modelscope import snapshot_download as s; s('m-a-p/MERT-v1-330M', local_dir=r'D:\sifeng-lora\models\ace\mert')"
.\venv\Scripts\python.exe -c "from modelscope import snapshot_download as s; s('utter-project/mHuBERT-147', local_dir=r'D:\sifeng-lora\models\ace\mhubert')"

# 数据集 + 改路径
cd D:\sifeng-lora\repo
..\venv\Scripts\python.exe convert2hf_dataset.py --data_dir "D:\sifeng-lora\train\data" --repeat_count 10 --output_name "D:\sifeng-lora\train\ds"
$f = "D:\sifeng-lora\repo\trainer.py"
(Get-Content $f -Raw).Replace('D:\AI\acestep\models_ms', 'D:\sifeng-lora\models\ace') | Set-Content $f -Encoding utf8 -NoNewline
```

## 7. 已修复的坑（不要重复踩）

1. `py3langid` 必须 **0.3.0**（0.4 删了 `from_pickled_model`）
2. torchaudio >= 2.9 的 load/save 需要 torchcodec → 代码已改为 soundfile
3. Windows 无 NCCL → 单卡策略已改 `strategy="auto"`
4. `plot_step` 里 `torch.distributed.get_rank()` 在单卡下崩 → 已加 `is_initialized()` 判断
5. MERT / mHuBERT 原本从 HuggingFace 拉取（被墙）→ 已改本地路径，走 ModelScope
6. 整首歌当样本时每个 step 要 3 分钟 → 已切成 30 秒片段

## 8. 已知限制

- 训练数据只有 57 段 30 秒（7 首曲子），LoRA r=16：风格会明显往司凤靠，但不会 100% 相同
- Windows 上多进程 dataloader 有可能崩，脚本会自动回退单进程（慢 5~8 倍）
