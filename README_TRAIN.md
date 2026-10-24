# 司凤风格 LoRA 训练包

本包用于在另一台带 GPU 的机器上训练 ACE-Step 的司凤风格 LoRA。
已包含：打过补丁的 ACE-Step 代码、57 段 30 秒司凤伴奏训练片段（含标签）、LoRA 配置、启动脚本。

## -1. 获取本包

私有仓库：https://github.com/werwerwerdanger/sifeng-lora-train

三种方式任选（国内网络打不开 GitHub 时，在 URL 前加 `https://gh-proxy.com/`）：

```bash
# A. 浏览器：登录 GitHub -> 仓库页面 -> Code -> Download ZIP（最省事，不需要 git）
# B. 已装 gh 并登录同一账号：
gh repo clone werwerwerdanger/sifeng-lora-train
# C. git + 个人访问令牌（私有仓库）：
git clone https://<TOKEN>@github.com/werwerwerdanger/sifeng-lora-train.git
```

解压/克隆后建议放在 `D:\sifeng-lora`（下面命令都按这个路径写，放别处就跟着改）。

## 0. 前置：目录结构

建议解压到与源机相同的路径，数据集里记录的是相对旧路径的绝对路径，
用脚本会**自动重建数据集**，所以路径可以不同，只要用下面的脚本生成数据集即可。

```
<任意目录>/
  repo/            # 已打补丁的 ACE-Step 代码
  train/
    data/          # 57 个 mp3 + _prompt.txt + _lyrics.txt
    sifeng_lora.json
  models/          # 需要单独准备（见第 2 步）
```

## 1. 准备 Python 环境（Python 3.12）

Windows:
```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  numpy diffusers "transformers==4.50.0" "accelerate==1.6.0" huggingface_hub loguru tqdm click `
  librosa soundfile datasets pypinyin "py3langid==0.3.0" num2words peft cutlet "fugashi[unidic-lite]" `
  hangul-romanize spacy matplotlib modelscope "pytorch_lightning==2.5.1" tensorboardX nnAudio
.\venv\Scripts\python.exe -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

Linux 同上，把 `.\venv\Scripts\python.exe` 换成 `venv/bin/python`。

> 注意：`py3langid` 必须是 0.3.0（0.4 删了 `from_pickled_model`）。
> torchaudio >= 2.9 需要 torchcodec；本包代码已改成用 soundfile，无需安装。

## 2. 下载模型（约 12.3 GB，走 ModelScope 最快）

```bash
python -c "from modelscope import snapshot_download as s; s('ACE-Step/ACE-Step-v1-3.5B', local_dir='models/ace')"
python -c "from modelscope import snapshot_download as s; s('m-a-p/MERT-v1-330M', local_dir='models/ace/mert')"
python -c "from modelscope import snapshot_download as s; s('utter-project/mHuBERT-147', local_dir='models/ace/mhubert')"
```

## 3. 重建数据集并改路径

```bash
cd repo
python convert2hf_dataset.py --data_dir "<训练包>/train/data" --repeat_count 10 --output_name "<训练包>/train/ds"
```

然后把 `trainer.py` 里的 MERT / mHuBERT 本地路径改成本机的 `models/ace/mert`、`models/ace/mhubert`
（搜 `D:\AI\acestep\models_ms` 替换即可）。

## 4. 启动训练

Windows: 双击 `train_windows.ps1`（或 `.\train_windows.ps1 300`）
Linux: `bash train_linux.sh 300`

参数说明：
- 第一个参数是训练步数，默认 100。8G 笔记本每步约 3 分 16 秒；12G 4070 桌面卡预计快 1.5~2 倍。
- 第二个参数是 dataloader worker 数，默认 8（i7-13700 的 16C/24T 足够）。Windows 版脚本在 worker 模式崩溃时**会自动回退到单进程**重跑。
- 多进程是训练速度的关键：本机 8G 笔记本 GPU 利用率只有 4%，就是因为单进程加载数据。
- 显存占用约 7.6G（bf16-mixed），12G 卡有余量。

## 5. 训练产物

- checkpoint：`train/logs/lightning_logs/<时间>_sifeng_lora/checkpoints/*.ckpt`
- 生成时用 `lora_name_or_path=<ckpt 路径>` + `lora_weight=1.0`

## 已知限制

- 数据只有 57 段 30 秒（7 首曲子），LoRA 容量 r=16，风格会往司凤靠但不会完全相同。
- 单卡 Windows 训练速度受 CPU 数据加载限制，Linux + num_workers 8 会快很多。
