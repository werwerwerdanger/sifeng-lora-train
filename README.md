# sifeng-tool

用多种神经网络方法处理音频 —— MIDI / 音频 到 8bit、古典 的音色转换工具。

## 结构

```
web/              前端 + Node 后端（纯原生模块，零 npm 依赖）
  index.html      转换工具界面
  server.js       上传 MIDI → 调用 Python 渲染 → 返回 WAV

scripts/          Python 渲染与处理脚本
  audio2midi.py   mp3/wav → 多轨 MIDI（Basic Pitch，逐轨转写）
  midi2nes.py     MIDI → NES 8bit（四声道合成 + 节奏范式）
  string_q4.py    MIDI → 弦乐四重奏（SoundFont 采样 + 定音鼓范式）
  nes_synth.py    NES 合成器内核（脉冲/三角/LFSR噪声）
  chiptune_stem.py 分离后逐轨 chiptune 化（波形整形+量化）
  sibilance.py    齿音移植（参考高频替换）
  master_eq.py    母带 EQ（peaking+shelf）
  mix.py          人声与伴奏混音（RMS 归一 + 侧链）
```

## 技术分层

| 层 | 技术 | 性质 |
|---|---|---|
| 分离 | BS-RoFormer / HTDemucs | 神经网络 |
| 转写 | Basic Pitch（ONNX） | 神经网络 |
| 音色转换 | RVC / ACE-Step | 神经网络 |
| 合成 | NES 四声道 / SF2 采样 | 确定性算法 |
| 母带 | EQ / 侧链 / 卷积混响 | DSP |

## 快速开始

```bash
# 后端（需要 Python 环境含 basic_pitch / pretty_midi / tinysoundfont）
cd web && node server.js 8899
# 打开 http://127.0.0.1:8899

# 命令行：MIDI → 8bit
python scripts/midi2nes.py --mid song.mid --out out.wav --drumpat euro

# 命令行：MIDI → 弦乐四重奏（默认已降密度，避免密集音符超出弦乐能力）
python scripts/string_q4.py --mid song.mid --out out.wav \
  --map "vln1=fife-1;vln2=guitar;vla=chord;vc=bass" --perc-pattern basic

# 命令行：音频 → MIDI（先分离再逐轨转写，准确率显著高于直接转混音）
python scripts/audio2midi.py --stems stems_dir --out song.mid
```

## 几个关键结论

- **转写必须逐轨**：直接转整段混音准确率 40–60%，分离后单轨 85–95%。
- **Basic Pitch 不量化时值**：时值唯一值 35–103 种，远超普通转写工具的 1–4 种。
- **古典改编不是换音色**：真实弦乐 0.3–1.5 音/秒，而流行歌吉他轨常有 4–6 音/秒、同时 5 音；
  必须降密度（单音化 + 速率上限）+ 补节奏乐器，否则听感像机器拉琴。
- **节奏乐器用范式生成**：按 BPM 铺满小节，不依赖 MIDI 自带鼓轨。
- **8bit 一条声道只承载一个声部**：多轨叠加会导致峰值叠加、掩蔽、crest 低于 5dB，听感"炸"。
