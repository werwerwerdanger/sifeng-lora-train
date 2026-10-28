# -*- coding: utf-8 -*-
"""任意 MIDI -> 弦乐四重奏 (SF2 采样音源)。

从 hito_q4_180.py 的思路简化: 那个版本是自己用模板编内声部,
这里直接吃 MIDI 已有的多轨, 按音区分配到四个弦乐声部。

轨道分配规则 (--map 可覆盖):
  高音轨 -> 小提琴1 (GM40)
  次高   -> 小提琴2 (GM40)
  中音   -> 中提琴  (GM41)
  低音   -> 大提琴  (GM42)

用法:
  python string_q4.py --mid x.mid --out q4.wav \
      --map "fife-1=vln1,electricguitar=vln2,guitarclean+1=vla,doublebass+2=vc"
"""
import argparse
import os

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import fftconvolve
import tinysoundfont as tsf

SR = 44100
SF2 = r"D:\sf\ext\SGM_V2_01_part2.sf2"

# 四声部: 名称 -> (通道, GM program)
VOICES = {
    "vln1": (0, 40),   # Violin
    "vln2": (1, 40),
    "vla":  (2, 41),   # Viola
    "vc":   (3, 42),   # Cello
    "timp": (4, 47),   # Timpani（定音鼓，给节奏骨架）
}



def thin(notes, mono=True, max_rate=None, dur=0.0, min_dur=0.06):
    """把不适合弦乐的密集音符变稀疏。

    弦乐四重奏的硬性约束: 每个声部同一时刻只能奏 1 个音(双音最多 2 个),
    而流行歌的吉他轨常有 4-6 音/秒、同时响 5 个 —— 直接映射必然"像机器拉琴"。

    mono      同簇(20ms 内)只留最高音 -> 复音降到 1~2
    max_rate  每秒最多 N 个音(按窗口保留时值最长的那个)
    min_dur   太短的音拉不出来, 补到最短时长
    """
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: n.start)
    if mono:
        out, cluster = [], []
        for n in notes:
            if cluster and n.start - cluster[-1].start > 0.02:
                out.append(max(cluster, key=lambda x: (x.pitch, x.end - x.start)))
                cluster = []
            cluster.append(n)
        if cluster:
            out.append(max(cluster, key=lambda x: (x.pitch, x.end - x.start)))
        notes = out
    if max_rate and dur > 0:
        win = 1.0 / max_rate
        buckets = {}
        for n in notes:
            k = int(n.start / win)
            cur = buckets.get(k)
            if cur is None or (n.end - n.start) > (cur.end - cur.start):
                buckets[k] = n
        notes = [buckets[k] for k in sorted(buckets)]
    fixed = []
    for n in notes:
        if n.end - n.start < min_dur:
            n = pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                 start=n.start, end=n.start + min_dur)
        fixed.append(n)
    return fixed



# 定音鼓节奏范式：按 4/4 小节铺满，描述"一小节里定音鼓落在哪"
# (拍位置, 力度)
TIMP_PATTERNS = {
    "basic":  [(0, 112), (2, 72)],                                  # 1、3拍(强/弱)
    "strong": [(0, 118)],                                            # 只 1 拍
    "half":   [(0, 110), (2, 96)],                                   # 两拍都重
    "roll":   [(i * 0.25, 60 + i * 6) for i in range(8)],            # 半拍滚奏
    "accent": [(0, 116), (1.5, 66), (2, 88), (3.5, 62)],             # 切分重音
    "none":   [],
}


def make_timp(name, dur, bpm):
    pat = TIMP_PATTERNS.get(name)
    if not pat:
        return []
    if bpm <= 0:
        bpm = 120.0
    spb = 60.0 / bpm
    out = []
    t = 0.0
    while t < dur:
        for beat, vel in pat:
            tt = t + beat * spb
            if tt < dur:
                out.append((tt, vel))
        t += spb * 4
    return out


def render(pm, mapping, out, gain=None, reverb=0.0, thinning=None, rate_limit=None,
           percussion_tracks=None, perc_rate=1.5, perc_gain=1.0, perc_pattern=None):
    thinning = thinning or {}
    rate_limit = rate_limit or {}
    dur = pm.get_end_time() + 3.0
    total = int(dur * SR)

    synth = tsf.Synth(samplerate=SR)
    sfid = synth.sfload(SF2)
    for name, (ch, prog) in VOICES.items():
        synth.program_select(ch, sfid, 0, prog)
    print(f"  program: vln1/vln2=40 vla=41 vc=42 timp=47")
    print(f"  soundfont OK  输出 {dur:.1f}s")

    # 收集事件
    ev = []
    by_name = {}
    for ins in pm.instruments:
        if ins.notes:
            by_name[(ins.name or "").lower()] = ins.notes

    for vname, track_names in mapping.items():
        ch = VOICES[vname][0]
        g = (gain or {}).get(vname, 1.0)
        raw = []
        for tn in track_names:
            raw += by_name.get(tn, [])
        n_raw = len(raw)
        raw = thin(raw, mono=thinning.get(vname, True),
                   max_rate=rate_limit.get(vname), dur=dur)
        cnt = 0
        for nt in raw:
            t0 = int(nt.start * SR)
            t1 = int(nt.end * SR)
            v = int(min(127, nt.velocity * g))
            if v < 1:
                continue
            ev.append((t0, 1, ch, nt.pitch, v))
            ev.append((t1, 0, ch, nt.pitch, 0))
            cnt += 1
        cut = f" ({n_raw}->{cnt})" if n_raw != cnt else ""
        print(f"  {vname:<5} <- {','.join(track_names):<30} {cnt:>5} 音{cut}  gain={g}")

    # 打击乐: 鼓轨映射到定音鼓, 音高收敛到 41-50（定音鼓实际音域）
    if perc_pattern and perc_pattern != "none":
        bpm = float(pm.estimate_tempo() or 120)
        tp = make_timp(perc_pattern, dur, bpm)
        ch = VOICES["timp"][0]
        for tt, vel in tp:
            v = int(min(120, vel * perc_gain))
            ev.append((int(tt * SR), 1, ch, 45, v))
            ev.append((int(tt * SR) + int(0.4 * SR), 0, ch, 45, 0))
        print(f"  timp  <- [pattern:{perc_pattern}]@{bpm:.0f}bpm   {len(tp):>5} 音  gain={perc_gain}")
    elif percussion_tracks:
        dp = []
        for tn in percussion_tracks:
            dp += by_name.get(tn, [])
        dp = thin(dp, mono=True, max_rate=perc_rate, dur=dur, min_dur=0.12)
        ch = VOICES["timp"][0]
        for nt in dp:
            p_ = 41 + (nt.pitch - 35) % 10
            v = int(min(120, nt.velocity * perc_gain))
            ev.append((int(nt.start * SR), 1, ch, p_, v))
            ev.append((int(nt.start * SR) + int(0.35 * SR), 0, ch, p_, 0))
        print(f"  timp  <- {','.join(percussion_tracks):<30} {len(dp):>5} 音  gain={perc_gain}")

    ev.sort(key=lambda x: x[0])
    print(f"  共 {len(ev)} 事件")

    buf = np.zeros(total, dtype=np.float32)
    BLOCK = 1024
    pos, i = 0, 0
    while pos < total:
        while i < len(ev) and ev[i][0] <= pos:
            _, kind, ch, p, v = ev[i]
            if kind:
                synth.noteon(ch, p, v)
            else:
                synth.noteoff(ch, p)
            i += 1
        n = min(BLOCK, total - pos)
        raw = np.asarray(synth.generate(n))
        arr = raw.view(np.float32) if raw.dtype == np.uint8 else raw.astype(np.float32).ravel()
        if arr.size == n * 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        buf[pos:pos + n] = arr[:n]
        pos += n

    if reverb > 0:
        buf = add_reverb(buf, SR, reverb)

    peak = float(np.abs(buf).max())
    if peak > 0:
        buf = buf / peak * 0.92
    sf.write(out, buf, SR)
    print(f"[输出] {out}")


def add_reverb(y, sr, amount=0.25):
    """简单卷积混响: 指数衰减噪声作 IR"""
    rng = np.random.default_rng(0)
    ir_len = int(1.6 * sr)
    t = np.arange(ir_len) / sr
    ir = rng.standard_normal(ir_len) * np.exp(-t / 0.45)
    ir[: int(0.012 * sr)] = 0          # 预延迟, 避免糊掉起音
    ir /= np.abs(ir).max()
    wet = fftconvolve(y, ir, mode="full")[: len(y)]
    return y * (1 - amount * 0.5) + wet * amount


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None,
                    help="vln1=a,b;vln2=c;vla=d;vc=e  (轨道名小写)")
    ap.add_argument("--gain", default="", help="vln1=0.9,vla=0.8 形式")
    ap.add_argument("--reverb", type=float, default=0.22)
    ap.add_argument("--rate", default="", help="每秒音符上限, 如 vln2=2,vla=1.5")
    ap.add_argument("--no-mono", action="store_true", help="不做单音化(保留复音)")
    ap.add_argument("--perc", default="", help="映射到定音鼓的鼓轨名, 逗号分隔")
    ap.add_argument("--perc-rate", type=float, default=1.5, help="定音鼓每秒上限")
    ap.add_argument("--perc-gain", type=float, default=1.0)
    ap.add_argument("--perc-pattern", default="",
                    help="定音鼓范式: basic/strong/half/roll/accent/none")
    a = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(a.mid)
    by_name = {(ins.name or "").lower(): ins.notes for ins in pm.instruments if ins.notes}
    print(f"[MIDI] {os.path.basename(a.mid)}  {pm.get_end_time():.1f}s  轨: {list(by_name.keys())}")

    if a.map:
        mapping = {}
        for part in a.map.split(";"):
            if "=" not in part:
                continue
            v, tracks = part.split("=", 1)
            mapping[v.strip()] = [x.strip().lower() for x in tracks.split(",") if x.strip()]
    else:
        # 自动: 按音域从高到低分配
        stats = []
        for nm, notes in by_name.items():
            ps = [n.pitch for n in notes]
            stats.append((np.mean(ps), nm))
        stats.sort(reverse=True)
        order = [nm for _, nm in stats]
        mapping = {"vln1": [order[0]], "vln2": [order[1]] if len(order) > 1 else [],
                   "vla": [order[2]] if len(order) > 2 else [],
                   "vc": [order[-1]] if len(order) > 3 else []}
        # 鼓轨排除
        mapping = {k: [x for x in v if "drum" not in x and "cymbal" not in x and "snare" not in x]
                   for k, v in mapping.items()}
        print(f"  自动映射: {mapping}")

    gain = {}
    for kv in a.gain.split(","):
        if "=" in kv:
            k, v = kv.split("=")
            gain[k.strip()] = float(v)

    rate_limit = {}
    for kv in a.rate.split(","):
        if "=" in kv:
            k, v = kv.split("=")
            rate_limit[k.strip()] = float(v)
    thinning = {v: (not a.no_mono) for v in mapping}
    perc = [x.strip().lower() for x in a.perc.split(",") if x.strip()]

    render(pm, mapping, a.out, gain, a.reverb, thinning, rate_limit,
           perc, a.perc_rate, a.perc_gain, a.perc_pattern)


if __name__ == "__main__":
    main()
