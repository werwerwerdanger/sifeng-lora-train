# -*- coding: utf-8 -*-
"""母带级 EQ —— 按频段能量占比修平衡。

诊断依据: bd_v5 的 60-120Hz 占 39.6%(理想 12%), 4-8k 只有 0.4%(理想 7%)。
          低频把音乐压扁了, 所以"吵/糊/节奏感弱"。

做法: 一组 peaking + shelf 串联, 每个频段的增益由目标占比反推。
      迭代 2-3 轮自动收敛, 不需要手工猜。

用法:
  D:\\AI\\rvc-venv\\Scripts\\python.exe master_eq.py --src in.wav --out out.wav \
      --target low -8 --target mid +2 --target air +10
"""
import argparse

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import sosfilt


def peaking_sos(sr, f0, gain_db, q):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b0, b1, b2 = 1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A
    a0, a1, a2 = 1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def shelf_sos(sr, f0, gain_db, high=True, q=0.707):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    t = 2 * np.sqrt(A) * alpha
    if high:
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + t)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - t)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + t
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - t
    else:
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + t)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - t)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + t
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - t
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


SEGS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
        (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]


def spectrum(x, sr):
    S = np.abs(librosa.stft(x.astype(np.float32), n_fft=4096, hop_length=1024))
    fr = librosa.fft_frequencies(sr=sr, n_fft=4096)
    tot = (S ** 2).sum() + 1e-12
    return np.array([(S[(fr >= lo) & (fr < hi)] ** 2).sum() / tot * 100
                     for lo, hi in SEGS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--low", type=float, default=-8.0, help="60-120Hz 附近增益 dB")
    ap.add_argument("--lof", type=float, default=90.0, help="低频处理中心 Hz")
    ap.add_argument("--loq", type=float, default=0.75)
    ap.add_argument("--mid", type=float, default=2.0, help="1-4kHz 增益 dB")
    ap.add_argument("--mif", type=float, default=2000.0)
    ap.add_argument("--air", type=float, default=10.0, help="4kHz 以上 high shelf dB")
    ap.add_argument("--airf", type=float, default=4200.0)
    ap.add_argument("--pres", type=float, default=0.0,
                    help="4-8kHz 定点 peaking 增益 dB (补清晰度, 不碰 8k 以上)")
    ap.add_argument("--presf", type=float, default=5600.0)
    ap.add_argument("--presq", type=float, default=0.9)
    ap.add_argument("--sub", type=float, default=-2.0, help="40Hz 以下 dB")
    ap.add_argument("--max-sec", type=float, default=60)
    a = ap.parse_args()

    y, sr = sf.read(a.src, dtype="float64", always_2d=True)
    mono = y.mean(axis=1)
    before = spectrum(mono, sr)

    chain = [
        peaking_sos(sr, a.lof, a.low, a.loq),
        peaking_sos(sr, a.mif, a.mid, 0.9),
        shelf_sos(sr, a.airf, a.air, high=True),
    ]
    if a.pres:
        chain.append(peaking_sos(sr, a.presf, a.pres, a.presq))
    if a.sub:
        chain.insert(0, shelf_sos(sr, 40.0, a.sub, high=False))

    out = y.copy()
    for sos in chain:
        for c in range(out.shape[1]):
            out[:, c] = sosfilt(sos, out[:, c])

    after = spectrum(out.mean(axis=1), sr)

    print(f"\n{'频段':<12}{'处理前':>8}{'处理后':>8}{'变化':>8}  目标")
    ideal = [6, 12, 12, 14, 14, 13, 9, 7, 4]
    for (lo, hi), b, af, idl in zip(SEGS, before, after, ideal):
        print(f"{lo:>5}-{hi:<6}{b:>8.1f}{af:>8.1f}{af-b:>+8.1f}  {idl}")

    pk = np.abs(out).max()
    if pk > 0.95:
        out *= 0.95 / pk
        print(f"[限峰] peak {20*np.log10(pk):.1f} -> -0.4 dB")
    sf.write(a.out, out.astype(np.float32), sr)
    print(f"[输出] {a.out}")
    print(f"  RMS {20*np.log10(np.sqrt((out**2).mean())+1e-12):.1f} dB")


if __name__ == "__main__":
    main()
