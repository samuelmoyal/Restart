#!/usr/bin/env python3
"""
Manual annotation of approach segments + crop boxes in the HPA landing films.

    PYTHONPATH=. python scripts/hpa_annotate.py download     # 720p H.264 preview copies of the channel
    PYTHONPATH=. python scripts/hpa_annotate.py serve        # then open http://localhost:8791

In the page: pick a video, scrub to the start of the approach and press "Start",
scrub to the end and press "End", click two opposite corners of the crop on the
picture, then "Add segment". Everything is saved at once to
``data/hpa/annotations.json``:

    {"<video id>": {"status": "done" | "skip" | "todo",
                    "segments": [{"t0": s, "t1": s, "crop": [x0, y0, x1, y1]}]}}

Crop coordinates are fractions of the frame, so the later extraction can run on a
higher-resolution download. Footage is used with the channel owner's permission;
``data/hpa`` is gitignored and must not be redistributed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("data/hpa")
PREVIEW = ROOT / "preview"
ANNOT = ROOT / "annotations.json"
INDEX = ROOT / "videos.json"
CHANNEL = "https://www.youtube.com/@HighPressureAviationFilms/videos"


def download(height: int, only: str | None, url: str = CHANNEL, tag: str = "") -> None:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        ["yt-dlp", "--js-runtimes", "node", "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s", url],
        capture_output=True, text=True, check=True).stdout
    videos = [dict(zip(("id", "duration", "title"), line.split("\t"))) for line in out.splitlines() if line.count("\t") == 2]
    for v in videos:
        v["source"] = url
        if tag:
            v["title"] = f"[{tag}] {v['title']}"
    # merge with videos already indexed from other channels / playlists
    known = json.loads(INDEX.read_text()) if INDEX.exists() else []
    ids = {v["id"] for v in videos}
    INDEX.write_text(json.dumps([v for v in known if v["id"] not in ids] + videos, indent=1, ensure_ascii=False))
    if only:
        videos = [v for v in videos if re.search(only, v["title"], re.I)]
    print(len(videos), "videos", flush=True)
    for k, v in enumerate(videos, 1):
        dst = PREVIEW / f"{v['id']}.mp4"
        if dst.exists():
            continue
        fmt = f"bv*[height<={height}][vcodec^=avc1]/bv*[height<={height}]"
        r = None
        for _ in range(2):
            r = subprocess.run(["yt-dlp", "-q", "--js-runtimes", "node", "-f", fmt, "--no-playlist",
                                "--merge-output-format", "mp4", "-o", str(dst.with_suffix(".%(ext)s")),
                                f"https://www.youtube.com/watch?v={v['id']}"], capture_output=True, text=True)
            if dst.exists():
                break
        size = f"{dst.stat().st_size / 1e6:.0f} MB" if dst.exists() else "FAILED " + (r.stderr.strip()[-160:] if r else "")
        print(f"[{k}/{len(videos)}] {v['id']} {v['title'][:50]} — {size}", flush=True)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>HPA annotate</title>
<style>
  :root { color-scheme: dark; --bg:#14171c; --panel:#1d2229; --line:#2e353f; --ink:#e8eaed; --mut:#9aa3ae; --acc:#eb6834; --ok:#1baf7a; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--ink); display:grid; grid-template-columns:320px 1fr; height:100vh; }
  #list { overflow:auto; border-right:1px solid var(--line); }
  #list header { position:sticky; top:0; background:var(--panel); padding:10px 12px; border-bottom:1px solid var(--line); font-weight:600; }
  .item { padding:8px 12px; border-bottom:1px solid var(--line); cursor:pointer; display:flex; gap:8px; align-items:flex-start; }
  .item:hover { background:#232a33; } .item.sel { background:#2a323d; }
  .dot { width:10px; height:10px; border-radius:50%; margin-top:4px; flex:none; background:#555d68; }
  .dot.done { background:var(--ok); } .dot.skip { background:#8a5a5a; }
  .item small { color:var(--mut); display:block; }
  main { overflow:auto; padding:14px 18px; }
  #stage { position:relative; width:100%; max-width:1180px; background:#000; }
  video { display:block; width:100%; }
  canvas { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
  .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:10px 0; max-width:1180px; }
  button { background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:7px 11px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--mut); } button.pri { background:var(--acc); border-color:var(--acc); color:#fff; font-weight:600; }
  button:disabled { opacity:.4; cursor:default; }
  .pill { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:6px 10px; font-variant-numeric:tabular-nums; }
  input[type=range] { flex:1; min-width:300px; accent-color:var(--acc); }
  table { border-collapse:collapse; width:100%; max-width:1180px; } td,th { padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; font-variant-numeric:tabular-nums; }
  th { color:var(--mut); font-weight:500; } kbd { background:#2a323d; border-radius:4px; padding:1px 5px; font:12px ui-monospace, monospace; }
  .hint { color:var(--mut); max-width:1180px; } label { color:var(--mut); display:flex; gap:6px; align-items:center; }
</style></head><body>
<div id="list"><header>Videos <span id="count" style="color:var(--mut);font-weight:400"></span></header><div id="items"></div></div>
<main>
  <h2 id="title" style="margin:0 0 8px">Pick a video</h2>
  <div id="stage"><video id="v" preload="metadata" playsinline muted></video><canvas id="c"></canvas></div>
  <div class="row">
    <button id="play">Play / pause</button>
    <input id="seek" type="range" min="0" max="1" step="0.01" value="0">
    <span class="pill" id="time">0.0 / 0.0 s</span>
    <label>speed <select id="rate"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
  </div>
  <div class="row">
    <button id="bStart">Start = now</button><span class="pill" id="pStart">start –</span>
    <button id="bEnd">End = now</button><span class="pill" id="pEnd">end –</span>
    <span class="pill" id="pCrop">crop – (click two corners)</span>
    <label><input type="checkbox" id="lock" checked> force 16:9</label>
    <button id="bClear">Clear crop</button>
    <button id="bAdd" class="pri" disabled>Add segment</button>
    <span id="need" style="font-weight:600"></span>
  </div>
  <table><thead><tr><th>#</th><th>start (s)</th><th>end (s)</th><th>length</th><th>crop x0,y0 → x1,y1</th><th></th></tr></thead><tbody id="segs"></tbody></table>
  <div class="row">
    <button id="bDone">Mark video done</button><button id="bSkip">No usable approach (skip)</button><button id="bNext">Next video →</button>
    <span id="saved" style="color:var(--mut)"></span>
  </div>
  <p class="hint">Keys: <kbd>space</kbd> play/pause · <kbd>←</kbd>/<kbd>→</kbd> ±1 s · <kbd>shift</kbd>+arrows ±10 s · <kbd>,</kbd>/<kbd>.</kbd> ±1 frame · <kbd>s</kbd> start · <kbd>e</kbd> end · <kbd>a</kbd> add segment · <kbd>n</kbd> next video.
  Draw the crop around the outside view only (no cockpit inset, captions or logo). The same crop is applied to the whole segment; add a second segment if the framing changes.</p>
</main>
<script>
const $ = id => document.getElementById(id);
const v = $('v'), c = $('c'), ctx = c.getContext('2d');
let videos = [], ann = {}, cur = null, t0 = null, t1 = null, pts = [], hover = null;
const fmt = t => t == null ? '–' : t.toFixed(1);

async function load() {
  videos = await (await fetch('/api/videos')).json();
  ann = await (await fetch('/api/annotations')).json();
  renderList();
  const first = videos.find(x => (ann[x.id]?.status ?? 'todo') === 'todo') || videos[0];
  if (first) select(first.id);
}
function renderList() {
  $('items').innerHTML = '';
  let done = 0;
  for (const x of videos) {
    const st = ann[x.id]?.status ?? 'todo'; if (st !== 'todo') done++;
    const d = document.createElement('div'); d.className = 'item' + (x.id === cur ? ' sel' : '');
    const n = ann[x.id]?.segments?.length ?? 0;
    d.innerHTML = `<span class="dot ${st}"></span><span>${x.title}<small>${Math.round(x.duration/60)} min · ${st}${n ? ' · ' + n + ' segment' + (n>1?'s':'') : ''}</small></span>`;
    d.onclick = () => select(x.id); $('items').appendChild(d);
  }
  $('count').textContent = `${done}/${videos.length} reviewed`;
}
function select(id) {
  cur = id; t0 = t1 = null; pts = [];
  const x = videos.find(y => y.id === id);
  $('title').textContent = x.title; v.src = '/video/' + id; v.playbackRate = +$('rate').value;
  renderList(); renderSegs(); refresh();
}
v.onloadedmetadata = () => { $('stage').style.aspectRatio = v.videoWidth + ' / ' + v.videoHeight; $('seek').max = v.duration; resize(); };
function resize() { const r = c.getBoundingClientRect(); c.width = r.width * devicePixelRatio; c.height = r.height * devicePixelRatio; draw(); }
window.onresize = resize;
v.ontimeupdate = () => { $('seek').value = v.currentTime; $('time').textContent = `${v.currentTime.toFixed(1)} / ${(v.duration||0).toFixed(1)} s`; };
$('seek').oninput = e => { v.currentTime = +e.target.value; };
$('play').onclick = () => v.paused ? v.play() : v.pause();
$('rate').onchange = e => v.playbackRate = +e.target.value;

function box() {
  if (pts.length < 1) return null;
  const a = pts[0], b = pts[1] ?? hover; if (!b) return null;
  let x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
  if ($('lock').checked && v.videoWidth) {         // keep 16:9 in pixels: shrink the longer side around the first click
    const ar = (16/9) * (v.videoHeight / v.videoWidth); let w = x1 - x0, h = y1 - y0;
    if (w / h > ar) w = h * ar; else h = w / ar;
    x0 = a[0] <= b[0] ? a[0] : a[0] - w; y0 = a[1] <= b[1] ? a[1] : a[1] - h; x1 = x0 + w; y1 = y0 + h;
  }
  const cl = t => Math.min(1, Math.max(0, t));
  return [cl(x0), cl(y0), cl(x1), cl(y1)];
}
function draw() {
  ctx.clearRect(0, 0, c.width, c.height);
  const W = c.width, H = c.height, b = box();
  for (const s of ann[cur]?.segments ?? []) {
    if (v.currentTime >= s.t0 && v.currentTime <= s.t1) { ctx.strokeStyle = '#1baf7a'; ctx.lineWidth = 2 * devicePixelRatio; ctx.strokeRect(s.crop[0]*W, s.crop[1]*H, (s.crop[2]-s.crop[0])*W, (s.crop[3]-s.crop[1])*H); }
  }
  if (b) {
    ctx.fillStyle = 'rgba(0,0,0,.45)'; ctx.fillRect(0, 0, W, b[1]*H); ctx.fillRect(0, b[3]*H, W, H - b[3]*H);
    ctx.fillRect(0, b[1]*H, b[0]*W, (b[3]-b[1])*H); ctx.fillRect(b[2]*W, b[1]*H, W - b[2]*W, (b[3]-b[1])*H);
    ctx.strokeStyle = '#eb6834'; ctx.lineWidth = 2 * devicePixelRatio; ctx.strokeRect(b[0]*W, b[1]*H, (b[2]-b[0])*W, (b[3]-b[1])*H);
  }
  requestAnimationFrame(() => {});
}
setInterval(draw, 120);
const pos = e => { const r = c.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
c.onmousemove = e => { hover = pos(e); if (pts.length === 1) draw(); };
c.onclick = e => { if (pts.length >= 2) pts = []; pts.push(pos(e)); refresh(); };
function refresh() {
  const b = pts.length === 2 ? box() : null;
  $('pStart').textContent = 'start ' + fmt(t0); $('pEnd').textContent = 'end ' + fmt(t1);
  $('pCrop').textContent = b ? `crop ${b.map(x => x.toFixed(3)).join(', ')}` : (pts.length === 1 ? 'crop: click the opposite corner' : 'crop – (click two corners)');
  const miss = [];
  if (t0 == null) miss.push('start time'); if (t1 == null) miss.push('end time');
  if (t0 != null && t1 != null && t1 <= t0) miss.push('end must be after start');
  if (!b) miss.push(pts.length === 1 ? 'second crop corner' : 'crop (click two corners on the picture)');
  $('bAdd').disabled = miss.length > 0;
  $('need').textContent = miss.length ? 'Missing: ' + miss.join(' · ') : 'Ready — press Add segment';
  $('need').style.color = miss.length ? '#f0a27e' : '#1baf7a';
  draw();
}
$('bStart').onclick = () => { t0 = v.currentTime; refresh(); };
$('bEnd').onclick = () => { t1 = v.currentTime; refresh(); };
$('bClear').onclick = () => { pts = []; refresh(); };
$('lock').onchange = refresh;
function entry() { return ann[cur] ??= {status: 'todo', segments: []}; }
$('bAdd').onclick = () => {
  const b = box(); if (!b) return;
  entry().segments.push({t0: +t0.toFixed(2), t1: +t1.toFixed(2), crop: b.map(x => +x.toFixed(4))});
  entry().status = 'done'; t0 = t1 = null; save(); renderSegs(); refresh();   // the crop stays, ready for another segment
};
function renderSegs() {
  const tb = $('segs'); tb.innerHTML = '';
  (ann[cur]?.segments ?? []).forEach((s, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i+1}</td><td>${s.t0.toFixed(1)}</td><td>${s.t1.toFixed(1)}</td><td>${(s.t1-s.t0).toFixed(1)} s</td><td>${s.crop.slice(0,2).map(x=>x.toFixed(3)).join(', ')} → ${s.crop.slice(2).map(x=>x.toFixed(3)).join(', ')}</td>`;
    const td = document.createElement('td');
    const go = document.createElement('button'); go.textContent = 'Go to'; go.onclick = () => { v.currentTime = s.t0; };
    const del = document.createElement('button'); del.textContent = 'Delete'; del.onclick = () => { ann[cur].segments.splice(i, 1); if (!ann[cur].segments.length) ann[cur].status = 'todo'; save(); renderSegs(); };
    td.append(go, ' ', del); tr.appendChild(td); tb.appendChild(tr);
  });
}
async function save() {
  $('saved').textContent = 'saving…';
  const r = await fetch('/api/annotations', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(ann)});
  $('saved').textContent = r.ok ? 'saved ' + new Date().toLocaleTimeString() : 'SAVE FAILED'; renderList();
}
$('bDone').onclick = () => { entry().status = 'done'; save(); };
$('bSkip').onclick = () => { entry().status = 'skip'; entry().segments = []; save(); renderSegs(); next(); };
function next() { const i = videos.findIndex(x => x.id === cur); const n = videos.slice(i + 1).find(x => (ann[x.id]?.status ?? 'todo') === 'todo') || videos[i + 1]; if (n) select(n.id); }
$('bNext').onclick = next;
document.addEventListener('click', e => { if (e.target.tagName === 'BUTTON') e.target.blur(); });
document.onkeydown = e => {
  if (e.target.tagName === 'SELECT') return;
  const fr = 1 / 30;
  if (e.key === ' ') { e.preventDefault(); $('play').click(); }
  else if (e.key === 'ArrowLeft') v.currentTime -= e.shiftKey ? 10 : 1;
  else if (e.key === 'ArrowRight') v.currentTime += e.shiftKey ? 10 : 1;
  else if (e.key === ',') { v.pause(); v.currentTime -= fr; } else if (e.key === '.') { v.pause(); v.currentTime += fr; }
  else if (e.key === 's') $('bStart').click(); else if (e.key === 'e') $('bEnd').click();
  else if (e.key === 'a') $('bAdd').click(); else if (e.key === 'n') next();
};
load();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/api/videos":
            meta = {v["id"]: v for v in json.loads(INDEX.read_text())} if INDEX.exists() else {}
            vids = []
            for p in sorted(PREVIEW.glob("*.mp4")):
                m = meta.get(p.stem, {})
                vids.append({"id": p.stem, "title": m.get("title", p.stem), "duration": float(m.get("duration") or 0)
                             if str(m.get("duration")) not in ("NA", "None") else 0.0})
            order = {vid: i for i, vid in enumerate(meta)}
            vids.sort(key=lambda v: order.get(v["id"], 1e9))
            return self._send(200, json.dumps(vids).encode(), "application/json")
        if self.path == "/api/annotations":
            return self._send(200, ANNOT.read_bytes() if ANNOT.exists() else b"{}", "application/json")
        m = re.fullmatch(r"/video/([A-Za-z0-9_-]{6,20})", self.path)
        if m and (PREVIEW / f"{m.group(1)}.mp4").exists():
            return self._video(PREVIEW / f"{m.group(1)}.mp4")
        self._send(404, b"not found", "text/plain")

    def _video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
        if rng:
            if rng.group(1):
                start = int(rng.group(1))
                end = int(rng.group(2)) if rng.group(2) else end
            elif rng.group(2):
                start = max(0, size - int(rng.group(2)))
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 18, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        if self.path != "/api/annotations":
            return self._send(404, b"not found", "text/plain")
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        tmp = ANNOT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(ANNOT)
        self._send(200, b"{}", "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch 720p H.264 preview copies")
    d.add_argument("--height", type=int, default=720)
    d.add_argument("--only", help="regex on the title, e.g. 'landing|approach'")
    d.add_argument("--url", default=CHANNEL, help="channel or playlist URL")
    d.add_argument("--tag", default="", help="prefix shown before the titles in the page")
    s = sub.add_parser("serve", help="run the annotation page")
    s.add_argument("--port", type=int, default=8791)
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.height, a.only, a.url, a.tag)
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        print(f"http://localhost:{a.port}  (Ctrl+C to stop) — annotations → {ANNOT}")
        ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
