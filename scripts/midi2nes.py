# -*- coding: utf-8 -*-
"""批量：任意 MIDI -> NES 8bit 风格。自动分析轨道并映射到四声道。

自动映射规则
  1. 鼓轨（名字含 drum/snare/cymbal/kick/hat/perc，或 MIDI 里标了 is_drum）-> Noise
  2. 其余按平均音高从高到低排：
       最高 -> Pulse1 (旋律)
       最低 -> 低音声道 (三角波/脉冲波)
       中间全部 -> Pulse2 (和弦/伴奏, 多轨叠加)
  3. 只有 2 个melodic轨时，高的给 Pulse1，低的给低音

用法
  # 单个
  python midi2nes.py --mid "x.mid" --out out/x_8bit.wav
  # 批量（目录里所有 .mid）
  python midi2nes.py --indir "D:/mymidi" --outdir out/batch
  # 覆盖自动映射
  python midi2nes.py --mid x.mid --out y.wav \
      --lead "fife-1" --p2 "electricguitar,guitarclean+1" --bass "doublebass+2" --drum "bassdrum+2,snaredrum+2"
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pretty_midi
import soundfile as sf

SR = 44100
DRUM_KW = ("drum", "snare", "cymbal", "kick", "hat", "perc", "timpani", "clap", "tom")


# ---------------------------------------------------------------- 合成器
def adsr(n, sr, a=0.005, d=0.03, s=0.7, r=0.04):
    env = np.full(n, s, dtype=np.float64)
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    na, nd, nr = min(na, n), min(nd, n), min(nr, n)
    if na:
        env[:na] = np.linspace(0, 1, na)
    if nd:
        env[na:na + nd] = np.linspace(1, s, min(nd, n - na))
    if nr:
        env[n - nr:] *= np.linspace(1, 0, nr)
    return env


def pulse(freq, n, sr, duty=0.5):
    t = np.arange(n) / sr
    ph = (t * freq) % 1.0
    return np.where(ph < duty, 1.0, -1.0)


def triangle(freq, n, sr):
    t = np.arange(n) / sr
    ph = (t * freq) % 1.0
    tri = 2 * np.abs(2 * ph - 1) - 1
    return np.round(tri * 15) / 15


def lfsr_noise(n, sr, rate=22050, short=False):
    taps = 6 if short else 1
    reg = 1
    hold = max(1, sr // rate)
    seq = np.empty(n // hold + 2, dtype=np.float64)
    for i in range(len(seq)):
        bit = (reg ^ (reg >> taps)) & 1
        reg = (reg >> 1) | (bit << 14)
        seq[i] = 1.0 if (reg & 1) else -1.0
    return np.repeat(seq, hold)[:n]


def hz(p):
    return 440.0 * 2 ** ((p - 69) / 12.0)


def render(notes, total_n, kind, gain, duty=0.5, vib=0.0, vib_hz=5.5):
    buf = np.zeros(total_n, dtype=np.float64)
    for nt in notes:
        i = int(nt.start * SR)
        n = int((nt.end - nt.start) * SR)
        if n <= 0 or i >= total_n:
            continue
        n = min(n, total_n - i)
        f = hz(nt.pitch)
        vel = nt.velocity / 127.0
        if kind == "pulse":
            if vib > 0 and n > int(0.18 * SR):
                t = np.arange(n) / SR
                mod = 2 ** (vib * np.sin(2 * np.pi * vib_hz * t) / 1200.0)
                ph = np.cumsum(f * mod) / SR % 1.0
                sig = np.where(ph < duty, 1.0, -1.0)
            else:
                sig = pulse(f, n, SR, duty)
            env = adsr(n, SR, r=min(0.04, (n / SR) * 0.5))
        elif kind == "triangle":
            sig = triangle(f, n, SR)
            env = adsr(n, SR, r=min(0.04, (n / SR) * 0.5))
        else:  # noise
            sig = lfsr_noise(n, SR, short=(nt.pitch % 2 == 1))
            t = np.arange(n) / SR
            tau = 0.02 + (nt.pitch - 36) * 0.004
            env = np.exp(-t / max(0.008, tau))
        buf[i:i + n] += sig * env * vel * gain
    return buf


# ---------------------------------------------------------------- 自动映射
def classify(pm):
    drums, mel = [], []
    for ins in pm.instruments:
        if not ins.notes:
            continue
        nm = (ins.name or "").lower()
        if ins.is_drum or any(k in nm for k in DRUM_KW):
            drums.append(ins)
        else:
            mel.append(ins)
    # 按平均音高降序
    mel.sort(key=lambda i: -float(np.mean([x.pitch for x in i.notes])))
    return drums, mel


def build_map(drums, mel):
    m = {"drum": [], "lead": [], "p2": [], "bass": []}
    for ins in drums:
        m["drum"].append((ins.name or "").lower())
    if len(mel) == 1:
        m["lead"] = [(mel[0].name or "").lower()]
    elif len(mel) == 2:
        m["lead"] = [(mel[0].name or "").lower()]
        m["bass"] = [(mel[1].name or "").lower()]
    else:
        m["lead"] = [(mel[0].name or "").lower()]
        m["bass"] = [(mel[-1].name or "").lower()]
        # p2 只有一条脉冲声道, 多轨叠加会糊成一片 -> 只取中间音数最多的那一轨
        mid = mel[1:-1]
        if mid:
            mid.sort(key=lambda i: -len(i.notes))
            m["p2"] = [(mid[0].name or "").lower()]
    return m


def shift(notes, semi):
    """八度移调: 音名不变, 只改音区"""
    if not semi:
        return notes
    return [pretty_midi.Note(velocity=x.velocity,
                             pitch=max(0, min(127, x.pitch + semi)),
                             start=x.start, end=x.end) for x in notes]



# ---------------------------------------------------------------- 节奏范式
# 每种范式描述"一个 4/4 小节里打击乐落在哪"，不依赖 MIDI 自带鼓轨。
# 值 = GM 打击乐音高: 35 kick / 38 snare / 42 closed hat / 46 open hat
KICK, SNARE, HAT, OHAT = 35, 38, 42, 46

PATTERNS = {
    # 名称: [(拍位置(0-3.99), 音高, 力度)]
    "four":   [(b, KICK, 108) for b in (0, 1, 2, 3)],
    "rock":   [(0, KICK, 110), (2, KICK, 104), (1, SNARE, 100), (3, SNARE, 100),
               (0.5, HAT, 62), (1.5, HAT, 58), (2.5, HAT, 62), (3.5, HAT, 58)],
    "euro":   [(b, KICK, 112) for b in (0, 1, 2, 3)] +
              [(b + 0.5, OHAT, 74) for b in (0, 1, 2, 3)],
    "offhat": [(b + 0.5, HAT, 70) for b in (0, 1, 2, 3)] +
              [(0, KICK, 96), (2, KICK, 92)],
    "hat8":   [(b, HAT, 68) for b in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5)],
    "half":   [(0, KICK, 110), (2.5, SNARE, 96)],
    "none":   [],
}


def make_pattern(name, dur, bpm, start=0.0):
    """按 BPM 把小节范式铺满整首, 返回 [(start_sec, pitch, vel)]"""
    pat = PATTERNS.get(name)
    if not pat:
        return []
    if bpm <= 0:
        bpm = 120.0
    spb = 60.0 / bpm            # 一拍秒数
    bar = spb * 4               # 一小节
    out = []
    t = start
    while t < dur:
        for beat, pitch, vel in pat:
            tt = t + beat * spb
            if tt < dur:
                out.append((tt, pitch, vel))
        t += bar
    return out


def collect(pm, names):
    by = {(i.name or "").lower(): i.notes for i in pm.instruments if i.notes}
    out = []
    for nm in names:
        out += by.get(nm, [])
    return out


def convert(mid_path, out_path, args, verbose=True):
    pm = pretty_midi.PrettyMIDI(mid_path)
    dur = pm.get_end_time()
    total_n = int(dur * SR) + SR

    drums, mel = classify(pm)
    auto = build_map(drums, mel)

    def pick(override, key):
        if override:
            return [x.strip().lower() for x in override.split(",") if x.strip()]
        return auto[key]

    lead_names = pick(args.lead, "lead")
    p2_names = pick(args.p2, "p2")
    bass_names = pick(args.bass, "bass")
    drum_names = pick(args.drum, "drum")

    lead = shift(collect(pm, lead_names), args.transpose_lead)
    p2 = collect(pm, p2_names)
    bass = shift(collect(pm, bass_names), args.transpose_bass)
    if args.drumpat and args.drumpat != "none":
        bpm = float(pm.estimate_tempo() or 120)
        pr = make_pattern(args.drumpat, dur, bpm)
        dr = [pretty_midi.Note(velocity=v, pitch=p, start=t, end=t + 0.05)
              for t, p, v in pr]
        drum_names = [f"[pattern:{args.drumpat}]@{bpm:.0f}bpm"]
    elif args.drumpat == "none":
        dr = []
        drum_names = ["[none]"]
    else:
        dr = collect(pm, drum_names)

    if verbose:
        print(f"\n[{os.path.basename(mid_path)}]  {dur:.1f}s  {pm.estimate_tempo():.0f}BPM  "
              f"{len(pm.instruments)}轨")
        print(f"  pulse1 <- {','.join(lead_names) or '(无)':<34} {len(lead):>5} 音")
        print(f"  pulse2 <- {','.join(p2_names) or '(无)':<34} {len(p2):>5} 音")
        print(f"  bass   <- {','.join(bass_names) or '(无)':<34} {len(bass):>5} 音")
        print(f"  noise  <- {','.join(drum_names) or '(无)':<34} {len(dr):>5} 音")

    out = np.zeros(total_n, dtype=np.float64)
    stems = {}
    if lead:
        stems["pulse1"] = render(lead, total_n, "pulse", args.gain_lead, args.duty_lead, vib=args.vib)
    if p2:
        stems["pulse2"] = render(p2, total_n, "pulse", args.gain_p2, args.duty_p2)
    if bass:
        stems["bass"] = render(bass, total_n, args.bass_kind, args.gain_bass, duty=0.5)
    if dr:
        stems["noise"] = render(dr, total_n, "noise", args.gain_drum)
    for v in stems.values():
        out += v

    if np.abs(out).max() < 1e-9:
        print("  !! 没有可渲染的轨道, 跳过")
        return None

    out = np.tanh(out * 0.95)
    out *= 0.86 / np.max(np.abs(out))
    if args.bits >= 2:
        lv = 2 ** (args.bits - 1)
        out = np.round(np.clip(out, -1, 1) * lv) / lv
    sf.write(out_path, out.astype(np.float32), SR)

    if verbose:
        S = np.abs(np.fft.rfft(out))
        fr = np.fft.rfftfreq(len(out), 1 / SR)
        tot = (S ** 2).sum() + 1e-12
        segs = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
                (1000, 2000), (2000, 4000), (4000, 8000)]
        spec = "  ".join(f"{lo}-{hi}:{(S[(fr>=lo)&(fr<hi)]**2).sum()/tot*100:.1f}"
                         for lo, hi in segs)
        print(f"  -> {out_path}")
        print(f"     频谱 {spec}")

    # 各声道包络(给前端画轨道波形): 每声道降采样成 N 点峰值
    if args.dump_stems:
        N = args.stem_points
        peaks = {}
        for name, buf in stems.items():
            if buf is None or len(buf) == 0:
                continue
            m = len(buf) // N * N
            if m < N:
                peaks[name] = [0.0] * N
                continue
            seg = np.abs(buf[:m]).reshape(N, -1)
            p = seg.max(axis=1)
            mx = p.max()
            peaks[name] = np.round(p / mx, 4).tolist() if mx > 0 else [0.0] * N
        meta = {"order": ["pulse1", "pulse2", "bass", "noise"], "points": N,
                "duration": round(dur, 2), "peaks": peaks}
        with open(args.dump_stems, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        if verbose:
            print(f"     声道包络 -> {args.dump_stems} ({list(peaks.keys())})")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", default=None, help="单个 MIDI")
    ap.add_argument("--indir", default=None, help="目录, 处理里面所有 .mid")
    ap.add_argument("--out", default=None, help="单文件输出路径")
    ap.add_argument("--outdir", default="out/batch", help="批量输出目录")
    ap.add_argument("--lead", default=None, help="覆盖旋律轨 (逗号分隔)")
    ap.add_argument("--p2", default=None, help="覆盖 pulse2 轨")
    ap.add_argument("--bass", default=None, help="覆盖低音轨")
    ap.add_argument("--drum", default=None, help="覆盖鼓轨")
    ap.add_argument("--bass-kind", default="pulse", choices=["pulse", "triangle"])
    ap.add_argument("--gain-lead", type=float, default=0.95)
    ap.add_argument("--gain-p2", type=float, default=0.32)
    ap.add_argument("--gain-bass", type=float, default=0.85)
    ap.add_argument("--gain-drum", type=float, default=0.40)
    ap.add_argument("--duty-lead", type=float, default=0.5)
    ap.add_argument("--duty-p2", type=float, default=0.25)
    ap.add_argument("--vib", type=float, default=16.0, help="旋律颤音 cents, 0=关")
    ap.add_argument("--bits", type=int, default=0, help="0=现代风格(推荐), 4=真NES")
    ap.add_argument("--transpose-bass", type=int, default=12,
                    help="低音移调, 默认+12(八度, 不改音名) 让能量进 120-250Hz")
    ap.add_argument("--transpose-lead", type=int, default=0)
    ap.add_argument("--drumpat", default="",
                    help="内置节奏范式(不依赖MIDI鼓轨): four/rock/euro/offhat/hat8/half")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dump-stems", default=None, help="输出各声道包络 JSON(给前端画波形)")
    ap.add_argument("--stem-points", type=int, default=720, help="包络点数")
    a = ap.parse_args()

    jobs = []
    if a.indir:
        os.makedirs(a.outdir, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(a.indir, "*.mid")) +
                        glob.glob(os.path.join(a.indir, "*.midi"))):
            jobs.append((p, os.path.join(a.outdir, os.path.splitext(os.path.basename(p))[0] + "_8bit.wav")))
    elif a.mid:
        out = a.out or os.path.join(a.outdir, os.path.splitext(os.path.basename(a.mid))[0] + "_8bit.wav")
        jobs.append((a.mid, out))
    else:
        raise SystemExit("需要 --mid 或 --indir")

    t0 = time.time()
    ok = 0
    for mid, out in jobs:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if a.skip_existing and os.path.exists(out):
            print(f"跳过已存在: {out}")
            continue
        try:
            mid_path = mid
            # 需要移调时先写到临时 MIDI
            r = convert(mid_path, out, a)
            if r:
                ok += 1
        except Exception as e:
            print(f"  !! 失败 {os.path.basename(mid)}: {e}")
    print(f"\n完成 {ok}/{len(jobs)} 个, 用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
