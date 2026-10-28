# -*- coding: utf-8 -*-
"""MIDI -> NES(FC) 四声道合成器。真方波, 不是滤波仿的。

NES 的硬件限制就是音色的来源:
  Pulse1  : 方波, 占空比 12.5/25/50%, 带 ADSR 和 vibrato
  Pulse2  : 同上 (两条脉冲轮换可以"同时"响两个音)
  Triangle: 三角波, 32 级量化, 无音量控制 (只有开/关)
  Noise   : 15-bit LFSR 噪声, 用于鼓

MIDI 轨道映射 (可按需改 --map):
  lead   -> pulse1
  arp    -> pulse2   (arp 本身就是快速轮换, 正好是 NES 的和弦手法)
  bass   -> triangle
  drums  -> noise
  piano/string -> 默认丢弃 (硬件没声道了; 用 --keep 可以混进 pulse2)

用法:
  python nes_synth.py --mid tg_style.mid --out nes.wav --bpm-from-midi
"""
import argparse
import os

import numpy as np
import pretty_midi
import soundfile as sf

SR = 44100


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
    # NES 三角波是 32 级阶梯
    return np.round(tri * 15) / 15


def lfsr_noise(n, sr, rate=22050, short=False):
    """NES 噪声声道: 15-bit LFSR, short 模式周期更短(更"金属")"""
    taps = 6 if short else 1
    reg = 1
    hold = max(1, sr // rate)
    seq = np.empty(n // hold + 2, dtype=np.float64)
    for i in range(len(seq)):
        bit = (reg ^ (reg >> taps)) & 1
        reg = (reg >> 1) | (bit << 14)
        seq[i] = 1.0 if (reg & 1) else -1.0
    return np.repeat(seq, hold)[:n]


def midi_to_hz(p):
    return 440.0 * 2 ** ((p - 69) / 12.0)


def render_track(notes, total_n, sr, kind, gain=1.0, duty=0.5,
                 vib_cents=0.0, vib_hz=5.5, decay=None):
    buf = np.zeros(total_n, dtype=np.float64)
    for nt in notes:
        i = int(nt.start * sr)
        n = int((nt.end - nt.start) * sr)
        if n <= 0 or i >= total_n:
            continue
        n = min(n, total_n - i)
        f = midi_to_hz(nt.pitch)
        vel = nt.velocity / 127.0

        if kind == "pulse":
            if vib_cents > 0 and n > int(0.18 * sr):
                t = np.arange(n) / sr
                mod = 2 ** (vib_cents * np.sin(2 * np.pi * vib_hz * t) / 1200.0)
                # 分块变速相位, 简化实现
                ph = np.cumsum(f * mod) / sr % 1.0
                sig = np.where(ph < duty, 1.0, -1.0)
            else:
                sig = pulse(f, n, sr, duty)
        elif kind == "triangle":
            sig = triangle(f, n, sr)
        else:
            sig = lfsr_noise(n, sr, short=(nt.pitch % 2 == 1))

        # 鼓用极短衰减, 音用 ADSR
        if kind == "noise":
            t = np.arange(n) / sr
            tau = 0.02 + (nt.pitch - 36) * 0.004
            env = np.exp(-t / max(0.008, tau))
        else:
            env = adsr(n, sr, r=min(0.04, (n / sr) * 0.5))
        buf[i:i + n] += sig * env * vel * gain
    return buf


def pick_lowest(notes, n_voice):
    """按时间聚簇, 每簇只留最低的 n_voice 个音"""
    notes = sorted(notes, key=lambda x: x.start)
    out = []
    cluster = []

    def flush(c):
        if not c:
            return
        c.sort(key=lambda x: x.pitch)
        out.extend(c[:n_voice])

    for nt in notes:
        if cluster and nt.start - cluster[-1].start > 0.12:
            flush(cluster)
            cluster = []
        cluster.append(nt)
    flush(cluster)
    return out


def humanize(notes, swing=0.0, jitter_ms=0.0, len_jit=0.0, seed=0):
    """把量化过的 MIDI 时值'松开': swing 延迟偶数位 + 随机抖动 + 时值伸缩"""
    rng = np.random.default_rng(seed)
    out = []
    for i, nt in enumerate(notes):
        st, en = nt.start, nt.end
        if swing > 0 and (i % 2 == 1):
            avg = (en - st)
            st = st + avg * swing * 0.5
        if jitter_ms > 0:
            st = st + rng.normal(0, jitter_ms / 1000.0)
        dur = en - nt.start
        if len_jit > 0:
            dur = dur * (1.0 + rng.uniform(-len_jit, len_jit))
        out.append(pretty_midi.Note(velocity=nt.velocity, pitch=nt.pitch,
                                    start=max(0.0, st), end=max(0.0, st) + max(0.02, dur)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gain-lead", type=float, default=0.55)
    ap.add_argument("--gain-arp", type=float, default=0.30)
    ap.add_argument("--gain-bass", type=float, default=0.62)
    ap.add_argument("--gain-drum", type=float, default=0.34)
    ap.add_argument("--duty-lead", type=float, default=0.25)
    ap.add_argument("--duty-arp", type=float, default=0.125)
    ap.add_argument("--vib", type=float, default=18.0, help="旋律颤音深度 cents")
    ap.add_argument("--keep", default="", help="额外保留的轨名(逗号分隔), 混进 pulse2")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="整体时间轴缩放 (MIDI 速度标错时用>, 如 1.446)")
    ap.add_argument("--swing", type=float, default=0.0,
                    help="摇摆量 0-0.5, 偶数位音符延后比例(按行内 index)")
    ap.add_argument("--human", type=float, default=0.0,
                    help="时序抖动 ms (标准差), 0=完全机械")
    ap.add_argument("--len-jit", type=float, default=0.0,
                    help="时值随机伸缩比例 0-0.5")
    ap.add_argument("--lead-track", default="lead,vocal,fife-1,fife",
                    help="旋律轨候选名, 逗号分隔, 按顺序找第一个有音符的")
    ap.add_argument("--p2-track", default="arp,electricguitar",
                    help="pulse2 用哪些轨, 逗号分隔(可多轨叠加)")
    ap.add_argument("--drum-track", default="drums,bassdrum+2,snaredrum+2,cymbal+2",
                    help="噪声声道用哪些轨, 逗号分隔")
    ap.add_argument("--transpose-lead", type=int, default=0, help="旋律移调半音")
    ap.add_argument("--transpose-arp", type=int, default=0, help="琶音移调半音")
    ap.add_argument("--transpose-bass", type=int, default=0, help="低音移调半音")
    ap.add_argument("--bits", type=int, default=0, help="0=不量化(现代风格), 4=真NES")
    ap.add_argument("--tri-kind", default="triangle", choices=["triangle", "pulse"],
                    help="低音用三角波(真NES)还是脉冲波(低频更厚)")
    ap.add_argument("--tri", default="bass", help="三角波用哪些轨, 逗号分隔 (bass,piano)")
    ap.add_argument("--tri-voice", type=int, default=0, help="三角波只取每轨最低的 N 个音, 0=全取")
    a = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(a.mid)
    dur = pm.get_end_time()
    total_n = int(dur * a.time_scale * SR) + SR
    keep = set(x.strip() for x in a.keep.split(",") if x.strip())

    by_name = {}
    for ins in pm.instruments:
        if ins.notes:
            by_name[(ins.name or "").lower()] = ins.notes
    if abs(a.time_scale - 1.0) > 1e-6:
        for k in list(by_name):
            by_name[k] = [pretty_midi.Note(velocity=x.velocity, pitch=x.pitch,
                                           start=x.start * a.time_scale,
                                           end=x.end * a.time_scale)
                          for x in by_name[k]]
        print(f"[时间轴] x{a.time_scale:.4f} -> 时长 {dur*a.time_scale:.1f}s")
    if a.human > 0 or a.swing > 0 or a.len_jit > 0:
        for k in list(by_name):
            by_name[k] = humanize(by_name[k], a.swing, a.human, a.len_jit)
        print(f"[人性化] swing={a.swing} jitter={a.human}ms len_jit={a.len_jit}")
    print(f"[MIDI] {os.path.basename(a.mid)}  {dur:.1f}s  轨: {list(by_name.keys())}")

    out = np.zeros(total_n, dtype=np.float64)

    # Pulse1 = lead
    lead = []
    for nm in [x.strip() for x in a.lead_track.split(",") if x.strip()]:
        if by_name.get(nm):
            lead = by_name[nm]
            print(f"  [映射] pulse1 <- {nm}")
            break
    if a.transpose_lead and lead:
        lead = [pretty_midi.Note(velocity=x.velocity, pitch=max(0, min(127, x.pitch + a.transpose_lead)),
                                 start=x.start, end=x.end) for x in lead]
    if lead:
        out += render_track(lead, total_n, SR, "pulse", a.gain_lead,
                            a.duty_lead, vib_cents=a.vib)
        print(f"  pulse1 <- lead        {len(lead)} 音  duty={a.duty_lead} vib={a.vib}c")

    # Pulse2 = arp (+ 可选 keep 轨)
    p2 = []
    for nm in [x.strip() for x in a.p2_track.split(",") if x.strip()]:
        if by_name.get(nm):
            p2 += by_name[nm]
    if a.transpose_arp and p2:
        p2 = [pretty_midi.Note(velocity=x.velocity, pitch=max(0, min(127, x.pitch + a.transpose_arp)),
                               start=x.start, end=x.end) for x in p2]
    for k in keep:
        p2 += by_name.get(k, [])
    if p2:
        out += render_track(p2, total_n, SR, "pulse", a.gain_arp, a.duty_arp)
        print(f"  pulse2 <- arp         {len(p2)} 音  duty={a.duty_arp}")

    # Triangle = bass
    bass = []
    for nm in [x.strip() for x in a.tri.split(",") if x.strip()]:
        src = by_name.get(nm, [])
        if nm != "bass" and a.tri_voice > 0:
            src = pick_lowest(src, a.tri_voice)   # 和弦轨只取根音
        bass += src
    if not bass:
        bass = by_name.get("bass", [])
    if a.transpose_bass and bass:
        bass = [pretty_midi.Note(velocity=x.velocity, pitch=max(0, min(127, x.pitch + a.transpose_bass)),
                                 start=x.start, end=x.end) for x in bass]
    if bass:
        out += render_track(bass, total_n, SR, a.tri_kind, a.gain_bass,
                            duty=0.5) if a.tri_kind == "pulse" else \
            out + render_track(bass, total_n, SR, "triangle", a.gain_bass)
        print(f"  tri    <- bass        {len(bass)} 音")

    # Noise = drums
    dr = []
    for nm in [x.strip() for x in a.drum_track.split(",") if x.strip()]:
        if by_name.get(nm):
            dr += by_name[nm]
    if dr:
        out += render_track(dr, total_n, SR, "noise", a.gain_drum)
        print(f"  noise  <- drums       {len(dr)} 音")

    if np.abs(out).max() < 1e-9:
        raise SystemExit("没有任何轨道被渲染, 检查 --mid 和轨道名")

    out = np.tanh(out * 1.2)                    # 轻度软削, 模拟混音台饱和
    out *= 0.92 / np.max(np.abs(out))
    if a.bits >= 2:
        lv = 2 ** (a.bits - 1)
        out = np.round(np.clip(out, -1, 1) * lv) / lv
        print(f"  [量化] {a.bits}-bit")
    sf.write(a.out, out.astype(np.float32), SR)

    S = np.abs(np.fft.rfft(out))
    fr = np.fft.rfftfreq(len(out), 1 / SR)
    tot = (S ** 2).sum() + 1e-12
    print(f"\n[输出] {a.out}  {dur:.1f}s")
    for lo, hi in [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
                   (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]:
        m = (fr >= lo) & (fr < hi)
        print(f"  {lo:>5}-{hi:<6}{(S[m]**2).sum()/tot*100:>7.2f}%")


if __name__ == "__main__":
    main()
