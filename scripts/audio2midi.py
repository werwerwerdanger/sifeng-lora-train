# -*- coding: utf-8 -*-
"""mp3/wav -> 多轨 MIDI（神经网络转写）。

管线: 分离后的 stems -> Basic Pitch 逐轨转写 -> 合并成多轨 MIDI
      Basic Pitch 输出的是连续时间, 不做网格量化, 所以时值不会被压平。

为什么必须逐轨:
  直接转整段混音时, 乐器泛音互相干扰、人声掩蔽其他声部, 准确率只有 40-60%。
  分轨后每轨是"相对干净的单音/少数音", 准确率能到 85%+。

用法:
  # 对已分离的 stems 目录
  python audio2midi.py --stems "sep4/htdemucs/no_vocals_128" --out out/track.mid
  # 对单个音频(不分离)
  python audio2midi.py --audio x.wav --out out/track.mid --name mix
"""
import argparse
import glob
import os
import time

import numpy as np
import pretty_midi

# Basic Pitch 内置 ONNX 模型(不需联网)
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

# 轨道名 -> (MIDI program, 是否鼓)
TRACK_META = {
    "vocals": (53, False),   # Voice Oohs（当人声旋律）
    "bass":   (33, False),   # Electric Bass (finger)
    "other":  (25, False),   # Acoustic Guitar (steel)
    "drums":  (0, True),
    "piano":  (0, False),
}


def transcribe(path, min_note_ms=60, min_pitch=24, max_pitch=108):
    """返回 pretty_midi.Instrument（单轨，未命名）"""
    _, midi, _ = predict(
        path,
        ICASSP_2022_MODEL_PATH,
        minimum_note_length=min_note_ms,
        minimum_frequency=440.0 * 2 ** ((min_pitch - 69) / 12),
        maximum_frequency=440.0 * 2 ** ((max_pitch - 69) / 12),
        multiple_pitch_bends=False,
    )
    if not midi.instruments:
        return None
    # predict 返回的可能是单轨，合并所有音符
    notes = [n for ins in midi.instruments for n in ins.notes]
    if not notes:
        return None
    inst = pretty_midi.Instrument(program=0)
    inst.notes = notes
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default=None, help="分离后的 stems 目录")
    ap.add_argument("--audio", default=None, help="单个音频文件")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-note", type=int, default=60, help="最短音符 ms")
    ap.add_argument("--only", default=None, help="只转写这些轨(逗号分隔)")
    ap.add_argument("--bpm", type=float, default=0, help="写入 tempo；0=不写")
    a = ap.parse_args()

    jobs = []
    if a.stems:
        for p in sorted(glob.glob(os.path.join(a.stems, "*.wav"))):
            name = os.path.splitext(os.path.basename(p))[0].lower()
            if a.only and name not in [x.strip().lower() for x in a.only.split(",")]:
                continue
            jobs.append((name, p))
    elif a.audio:
        jobs.append((os.path.splitext(os.path.basename(a.audio))[0].lower(), a.audio))
    else:
        raise SystemExit("需要 --stems 或 --audio")

    print(f"[Basic Pitch] 模型 {ICASSP_2022_MODEL_PATH.name}  待转写 {len(jobs)} 轨")

    out_pm = pretty_midi.PrettyMIDI(initial_tempo=a.bpm) if a.bpm else pretty_midi.PrettyMIDI()
    total_notes = 0
    t_all = time.time()

    for name, path in jobs:
        t0 = time.time()
        try:
            inst = transcribe(path, a.min_note)
        except Exception as e:
            print(f"  {name:<10} 失败: {e}")
            continue
        if inst is None:
            print(f"  {name:<10} 无音符, 跳过")
            continue
        prog, is_drum = TRACK_META.get(name, (0, False))
        inst.program = prog
        inst.is_drum = is_drum
        inst.name = name
        out_pm.instruments.append(inst)

        d = np.array([n.end - n.start for n in inst.notes])
        ps = [n.pitch for n in inst.notes]
        print(f"  {name:<10} {len(inst.notes):>5} 音  音域 {min(ps)}-{max(ps)}  "
              f"时值 {np.median(d)*1000:.0f}ms 中位  唯一值 {len(np.unique(np.round(d,3)))}  "
              f"({time.time()-t0:.1f}s)")
        total_notes += len(inst.notes)

    if total_notes == 0:
        raise SystemExit("没有转写出任何音符")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out_pm.write(a.out)
    print(f"\n[输出] {a.out}  {len(out_pm.instruments)} 轨  {total_notes} 音  "
          f"时长 {out_pm.get_end_time():.1f}s  用时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
