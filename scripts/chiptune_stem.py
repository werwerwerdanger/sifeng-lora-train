# -*- coding: utf-8 -*-
"""分离后逐轨做 chiptune 化 —— 验证「分离完再丢信息」够不够。

三条处理串联, 作用完全不同:
  1. squarify  波形整形: 把平滑波形"推"成方波状（这才是 chiptune 的核心）
  2. downsample 降采样: 砍掉高频, 产生颗粒感
  3. bitcrush  量化: 4-bit 音量台阶, 复古噪点

只做 2+3 就是 lo-fi；必须加 1 才有"方波音色"。

用法:
  python chiptune_stem.py --stems sep4/htdemucs/no_vocals_128 --out out/8bit_try.wav
"""
import argparse
import glob
import os

import librosa
import numpy as np
import soundfile as sf


def squarify(y, drive=6.0):
    """高增益 tanh 软削波 -> 波形趋近方波。drive 越大越"方"。"""
    return np.tanh(y * drive) / np.tanh(drive)


def downsample(y, sr, target=11025):
    lo = librosa.resample(y, orig_sr=sr, target_sr=target)
    return librosa.resample(lo, orig_sr=target, target_sr=sr)


def bitcrush(y, bits=4):
    lv = 2 ** (bits - 1)
    return np.round(np.clip(y, -1, 1) * lv) / lv


def gate(y, sr, thresh_db=-26.0, rel_ms=45):
    """能量门限: 切掉混响尾音, 让音符"干掉"。
    NES 没有混响, 每个音就是 square + ADSR, 尾部是硬衰减的。"""
    from scipy.ndimage import uniform_filter1d
    env = librosa.feature.rms(y=y, frame_length=256, hop_length=64)[0]
    edb = 20 * np.log10(env + 1e-9)
    peak = np.percentile(edb, 99)
    open_mask = (edb > peak + thresh_db).astype(np.float64)
    # 短释放, 避免切出咔哒
    n = max(1, int(rel_ms / 1000 * sr / 64))
    g = uniform_filter1d(open_mask, size=n, mode="nearest")
    gi = np.interp(np.arange(len(y)), np.arange(len(g)) * 64, g)
    return y * gi


def process(y, sr, drive, target_sr, bits, hp, gate_on):
    y = y / (np.abs(y).max() + 1e-9)
    if gate_on:
        y = gate(y, sr)                     # 先去混响尾, 再做整形
    y = squarify(y, drive)                  # 推成方波
    if hp:
        from scipy.signal import butter, sosfilt
        y = sosfilt(butter(4, hp, "hp", fs=sr, output="sos"), y)  # 去 sub bass
    y = downsample(y, sr, target_sr)        # 削高频
    y = bitcrush(y, bits)                   # 音量台阶
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True, help="含 drums/bass/other.wav 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drive", type=float, default=6.0, help="波形整形强度, 越大越方")
    ap.add_argument("--target-sr", type=int, default=11025, help="降采样目标")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--hp", type=float, default=55.0, help="高通(去 sub), 0=关")
    ap.add_argument("--gate", action="store_true", default=True, help="门限去混响尾")
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--balance", default="drums=0.55,bass=0.5,other=0.85",
                    help="各声部电平")
    a = ap.parse_args()

    bal = {}
    for kv in a.balance.split(","):
        k, v = kv.split("=")
        bal[k.strip()] = float(v)

    mix = None
    sr = None
    print(f"[参数] drive={a.drive} target_sr={a.target_sr} bits={a.bits} hp={a.hp}")
    for name in ("drums", "bass", "other"):
        p = os.path.join(a.stems, name + ".wav")
        if not os.path.exists(p):
            print(f"  跳过 {name}: 不存在")
            continue
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        # 各自先归一, 避免某一轨主导
        y = y / (np.abs(y).max() + 1e-9)
        y = process(y, sr, a.drive, a.target_sr, a.bits, a.hp, a.gate)
        g = bal.get(name, 0.7)
        print(f"  {name:<6} 整形+降采样+量化完成  增益 {g}")
        mix = y * g if mix is None else mix + y * g

    # 4-bit 音量台阶应该在总线上也做一次, 才是"整机只有 16 级音量"
    mix = bitcrush(mix / (np.abs(mix).max() + 1e-9) * 0.92, a.bits)
    sf.write(a.out, mix, sr)

    S = np.abs(librosa.stft(mix.astype(np.float32), n_fft=4096, hop_length=1024))
    fr = librosa.fft_frequencies(sr=sr, n_fft=4096)
    tot = (S ** 2).sum() + 1e-12
    print(f"\n[输出] {a.out}")
    for lo, hi in [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
                   (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]:
        v = (S[(fr >= lo) & (fr < hi)] ** 2).sum() / tot * 100
        print(f"  {lo:>5}-{hi:<6}{v:>7.2f}%")


if __name__ == "__main__":
    main()
