// MIDI -> 8bit / 古典 转换服务。纯 Node 原生模块，无需 npm install。
// 启动: node server.js [port]
const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = Number(process.argv[2] || 8899);
const ROOT = __dirname;
const UP = path.join(ROOT, 'uploads');
const OUT = path.join(ROOT, 'outputs');
const PY = 'D:\\AI\\rvc-venv\\Scripts\\python.exe';
const ACESTEP = 'D:\\AI\\acestep';

[UP, OUT].forEach(d => fs.mkdirSync(d, { recursive: true }));

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.mid': 'audio/midi',
  '.json': 'application/json; charset=utf-8',
};

function send(res, code, body, mime) {
  res.writeHead(code, {
    'Content-Type': mime || 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
    'Cache-Control': 'no-store',
  });
  res.end(body);
}

// 静态文件
function serveStatic(req, res) {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/index.html';
  const file = path.join(ROOT, p);
  if (!file.startsWith(ROOT)) return send(res, 403, '{"error":"forbidden"}');
  fs.readFile(file, (err, data) => {
    if (err) return send(res, 404, '{"error":"not found"}');
    send(res, 200, data, MIME[path.extname(file).toLowerCase()] || 'application/octet-stream');
  });
}

// 解析 python 输出里的指标
function parseStats(stdout) {
  const st = {};
  const m = stdout.match(/频谱\s+((?:\d+-\d+:[\d.]+(?:\s+|$))+)/);
  if (m) {
    st.spectrum = {};
    for (const kv of m[1].trim().split(/\s+/)) {
      const [band, val] = kv.split(':');
      if (band && val) st.spectrum[band] = Number(val);
    }
  }
  const dur = stdout.match(/(\d+\.\d+)s\s+\d+BPM/);
  if (dur) st.duration = Number(dur[1]);
  const trk = [...stdout.matchAll(/(pulse[12]|bass|noise|vln[12]|vla|vc)\s*<-\s*(\S+)(?:\s|$)/g)];
  if (trk.length) st.mapping = trk.map(x => `${x[1]} ← ${x[2]}`);
  return st;
}

function runPython(args, cb) {
  const t0 = Date.now();
  const child = spawn(PY, args, { cwd: ACESTEP, windowsHide: true });
  let out = '', err = '';
  child.stdout.on('data', d => { out += d.toString('utf8'); });
  child.stderr.on('data', d => { err += d.toString('utf8'); });
  child.on('close', code => cb(code, out, err, Date.now() - t0));
  child.on('error', e => cb(-1, out, String(e), Date.now() - t0));
}

function handleConvert(req, res) {
  let raw = '';
  req.on('data', c => {
    raw += c;
    if (raw.length > 120 * 1024 * 1024) { req.destroy(); }
  });
  req.on('end', () => {
    let body;
    try { body = JSON.parse(raw); } catch (e) { return send(res, 400, '{"error":"bad json"}'); }

    const { filename, data, mode, opts = {} } = body;
    if (!filename || !data) return send(res, 400, '{"error":"missing filename/data"}');

    const stamp = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const safe = filename.replace(/[^\w\u4e00-\u9fff.-]/g, '_').slice(-40);
    const midPath = path.join(UP, `${stamp}_${safe}`);
    const outName = `${stamp}_${mode}.wav`;
    const outPath = path.join(OUT, outName);

    try {
      fs.writeFileSync(midPath, Buffer.from(data, 'base64'));
    } catch (e) {
      return send(res, 500, JSON.stringify({ error: 'save failed: ' + e.message }));
    }

    let args;
    const stemPath = path.join(OUT, `${stamp}_stems.json`);
    if (mode === 'a2m') {
      // 音频 → MIDI → 顺便渲染一份 8bit 试听
      const midOut = path.join(OUT, `${stamp}_a2m.mid`);
      const minNote = String(opts.minNote ?? 60);
      const a1 = ['audio2midi.py', '--audio', midPath, '--out', midOut, '--min-note', minNote];
      console.log('[a2m] 转写中…');
      return runPython(a1, (c1, out1, err1, ms1) => {
        if (c1 !== 0 || !fs.existsSync(midOut)) {
          console.error('[a2m fail]', c1, err1.slice(-500));
          return send(res, 500, JSON.stringify({ error: 'transcription failed', log: (out1 + err1).slice(-1400) }));
        }
        const a2 = ['midi2nes.py', '--mid', midOut, '--out', outPath,
                    '--transpose-bass', '0', '--bits', '0'];
        if (opts.drumPat) a2.push('--drumpat', opts.drumPat);
        console.log('[a2m] 渲染试听…');
        runPython(a2, (c2, out2, err2, ms2) => {
          if (c2 !== 0 || !fs.existsSync(outPath)) {
            return send(res, 500, JSON.stringify({ error: 'render failed', log: (out2 + err2).slice(-1400) }));
          }
          // 解析转写统计
          const stats = { midi: {} };
          const rows = [...out1.matchAll(/(\S+)\s+(\d+)\s*音\s+音域\s*(\d+)-(\d+)\s+时值\s*(\d+)ms\s*中位\s+唯一值\s+(\d+)/g)];
          if (rows.length) {
            stats.midi.tracks = rows.map(m => ({
              name: m[1], notes: +m[2], lo: +m[3], hi: +m[4], medMs: +m[5], uniq: +m[6],
            }));
            stats.midi.totalNotes = stats.midi.tracks.reduce((a, x) => a + x.notes, 0);
          }
          const d = out1.match(/时长s+([d.]+)s/);
          if (d) stats.midi.duration = +d[1];
          stats.spectrum = parseStats(out2).spectrum;
          const size = fs.statSync(outPath).size;
          const midiSize = fs.statSync(midOut).size;
          console.log(`[a2m ok] ${(stats.midi.totalNotes||0)} 音  ${((ms1+ms2)/1000).toFixed(1)}s`);
          send(res, 200, JSON.stringify({
            ok: true,
            url: '/outputs/' + encodeURIComponent(path.basename(outPath)),
            midiUrl: '/outputs/' + encodeURIComponent(path.basename(midOut)),
            size, midiSize, ms: ms1 + ms2,
            stats,
            log: (out1 + '\n' + out2).slice(-900),
          }));
        });
      });
    }
    if (mode === 'nes') {
      args = ['midi2nes.py', '--mid', midPath, '--out', outPath,
        '--bits', String(opts.bits ?? 0),
        '--transpose-bass', String(opts.transposeBass ?? 12),
        '--vib', String(opts.vib ?? 16),
        '--bass-kind', opts.bassKind || 'pulse',
        '--dump-stems', stemPath];
      if (opts.drumPat) args.push('--drumpat', opts.drumPat);
    } else if (mode === 'q4') {
      args = ['string_q4.py', '--mid', midPath, '--out', outPath,
        '--reverb', String(opts.reverb ?? 0.25)];
      if (opts.map) args.push('--map', opts.map);
      if (opts.percPat) args.push('--perc-pattern', opts.percPat);
      args.push('--rate', opts.rate || 'vln1=2.2,vln2=1.8,vla=1.5,vc=1.8');
    } else {
      return send(res, 400, '{"error":"unknown mode"}');
    }

    console.log(`[convert] ${safe}  mode=${mode}  ${args.slice(0, 3).join(' ')}`);
    runPython(args, (code, stdout, stderr, ms) => {
      if (code !== 0 || !fs.existsSync(outPath)) {
        console.error('[fail]', code, stderr.slice(-600));
        return send(res, 500, JSON.stringify({
          error: 'conversion failed',
          code,
          log: (stdout + stderr).slice(-1200),
        }));
      }
      const size = fs.statSync(outPath).size;
      console.log(`[ok] ${outName}  ${(size / 1e6).toFixed(1)}MB  ${ms}ms`);
      let stems = null;
      try {
        if (fs.existsSync(stemPath)) stems = JSON.parse(fs.readFileSync(stemPath, 'utf8'));
      } catch (e) { /* 包络可选 */ }
      send(res, 200, JSON.stringify({
        ok: true,
        url: '/outputs/' + encodeURIComponent(outName),
        size,
        ms,
        stats: { ...parseStats(stdout), stems },
        log: stdout.slice(-900),
      }));
    });
  });
}

const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') return send(res, 204, '');
  if (req.method === 'POST' && req.url === '/api/convert') return handleConvert(req, res);
  if (req.method === 'GET') return serveStatic(req, res);
  send(res, 405, '{"error":"method not allowed"}');
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`\n  MIDI 转换服务已启动`);
  console.log(`  →  http://127.0.0.1:${PORT}\n`);
});
