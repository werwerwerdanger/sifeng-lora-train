# -*- coding: utf-8 -*-
"""人声 + 伴奏 混音。

要点(都是踩过的坑):
  1. 人声必须先做 RMS 归一, 否则会小到听不见(之前差 15dB)
  2. 人声出现时伴奏低频让位(侧链), 否则人声被埋
  3. 不加 tanh 硬饱和, 会糊
  4. 给 headroom, 最后只压真超限的峰值

用法:
  D:\\AI\\rvc-venv\\Scripts\\python.exe mix.py --vocal v.wav --bed b.wav --out mix.wav \
      --vocal-db -2 --duck 0.82
"""
import argparse

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


def db(x):
    return 20 * np.log10(x + 1e-12)


def rms(x):
    return np.sqrt((x ** 2).mean()) + 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True)
    ap.add_argument("--bed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocal-db", type=float, default=-2.0,
                    help="人声 RMS 相对伴奏的 dB (0=等响, -2=略低, +2=略高)")
    ap.add_argument("--duck", type=float, default=0.82, help="侧链深度 0.5=压一半")
    ap.add_argument("--duck-rel", type=float, default=180, help="侧链释放 ms")
    ap.add_argument("--duck-f", type=float, default=320, help="侧链作用上限 Hz")
    ap.add_argument("--vocal-hp", type=float, default=75, help="人声高通 Hz")
    ap.add_argument("--max-sec", type=float, default=70)
    a = ap.parse_args()

    v, sv = librosa.load(a.vocal, sr=None, mono=True, duration=a.max_sec)
    b, sb = librosa.load(a.bed, sr=None, mono=True, duration=a.max_sec)
    if sv != sb:
        v = librosa.resample(v, orig_sr=sv, target_sr=sb)
    n = min(len(v), len(b))
    v, b = v[:n].astype(np.float64), b[:n].astype(np.float64)
    sr = sb

    # 人声高通, 去掉 RVC 留下的低频轰
    v = sosfilt(butter(4, a.vocal_hp, "hp", fs=sr, output="sos"), v)

    # 电平: 以伴奏 RMS 为基准
    b_rms = rms(b)
    v_rms = rms(v)
    target = b_rms * (10 ** (a.vocal_db / 20.0))
    v = v * (target / v_rms)
    print(f"[电平] 伴奏 {db(b_rms):.1f}dB  人声 {db(v_rms):.1f} -> {db(target):.1f}dB "
          f"({a.vocal_db:+.1f}dB)")

    # 侧链: 用人声包络驱动伴奏低频。攻击立即, 释放平滑
    hop = 256
    env_v = librosa.feature.rms(y=v, frame_length=1024, hop_length=hop)[0]
    env_v = env_v / (env_v.max() + 1e-12)
    need = a.duck + (1 - a.duck) * (1 - np.clip(env_v / 0.25, 0, 1))
    rel = max(1e-6, 1.0 / max(1, int(a.duck_rel / 1000 * sr / hop)))
    g = np.ones_like(need)
    for i in range(1, len(need)):
        if need[i] < g[i - 1]:
            g[i] = need[i]                       # 攻击: 立即压下
        else:
            g[i] = min(need[i], g[i - 1] + rel)  # 释放: 慢慢回来
    # 上采样到采样点
    t = np.arange(len(g)) * hop
    gi = np.interp(np.arange(n), t, g)[:n]

    sos_lo = butter(4, a.duck_f, "lp", fs=sr, output="sos")
    sos_hi = butter(4, a.duck_f, "hp", fs=sr, output="sos")
    b_ducked = sosfilt(sos_lo, b) * gi + sosfilt(sos_hi, b)

    mix = b_ducked + v
    mix_rms = rms(mix)
    print(f"[侧链] 深度 {a.duck}  释放 {a.duck_rel}ms  "
          f"作用 {a.duck_f}Hz 以下  平均增益 {db(gi.mean()):.1f}dB")
    print(f"[混音] RMS {db(mix_rms):.1f}dB  peak {db(np.abs(mix).max()):.1f}dB")

    pk = np.abs(mix).max()
    if pk > 0.92:
        mix *= 0.92 / pk
        print(f"[限峰] -> {db(0.92):.1f}dB")

    sf.write(a.out, mix.astype(np.float32), sr)
    print(f"[输出] {a.out}  {n/sr:.1f}s")


if __name__ == "__main__":
    main()
