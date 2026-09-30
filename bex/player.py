"""The Explorer's audio player: one timeline, in recording time, under the spectrogram.

Streamlit's own player counts from zero for whatever clip it is given, so the
60 s under a spectrogram labelled 1920–1980 s plays as 0:00–1:00, and lining a
sound up with what you are looking at meant doing the subtraction in your head.
This player draws its own timeline in recording time, with the same ticks as
the figure above it, inset by the figure's own margins so the two line up. It
shows the inspected window as a band, and moves a playhead across as it plays.
It also plays just that window.

Audio is sent as MP3: about 0.5 MB a minute where 16-bit WAV would be ~4 MB,
and unlike OGG every browser plays it. MP3 adds a few tens of milliseconds of
encoder delay at the start, far below anything visible at these spans.
"""
from __future__ import annotations

import base64
import html
import io
import json

import numpy as np
import soundfile as sf


def encode_mp3(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(y, dtype=np.float32), sr, format="MP3",
             subtype="MPEG_LAYER_III")
    return buf.getvalue()


def player_html(
    audio: bytes,
    t0: float,
    t1: float,
    clip_start: float,
    window: tuple[float, float] | None,
    left_frac: float,
    right_frac: float,
    ticks: list[float],
) -> str:
    """The player as a self-contained HTML snippet.

    `t0`/`t1` is the visible span in recording time and `clip_start` the
    recording time of the audio's first sample. `left_frac` / `right_frac` are
    where the figure's plotting area starts and ends as fractions of its width
    (matplotlib's axes box), so the timeline sits exactly under the time axis.
    """
    span = max(t1 - t0, 1e-9)
    pct = lambda t: 100.0 * (min(max(t, t0), t1) - t0) / span
    tick_html = "".join(
        f'<span class="tk" style="left:{pct(t):.3f}%">{t:g}</span>'
        for t in ticks if t0 <= t <= t1)
    win_html, win_btn = "", ""
    if window:
        w0, w1 = window
        win_html = (f'<div class="win" style="left:{pct(w0):.3f}%;'
                    f'width:{pct(w1) - pct(w0):.3f}%"></div>')
        win_btn = (f'<button id="pw" title="Play only the inspected window">'
                   f'▶ window {w0:g}–{w1:g} s</button>')
    cfg = json.dumps({"t0": t0, "t1": t1, "c0": clip_start,
                      "w": list(window) if window else None})
    src = "data:audio/mpeg;base64," + base64.b64encode(audio).decode()
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        font-size:12px; color:#6b6a63; background:transparent; }}
#wrap {{ padding:0 {100 * (1 - right_frac):.3f}% 0 {100 * left_frac:.3f}%; }}
#bar {{ position:relative; height:22px; background:#ecebe6; border-radius:4px;
        cursor:pointer; }}
.win {{ position:absolute; top:0; bottom:0; background:#2a78d633;
        border:1.5px solid #2a78d6; border-radius:3px; box-sizing:border-box; }}
#head {{ position:absolute; top:-3px; bottom:-3px; width:2px; margin-left:-1px;
         background:#d0021b; left:0; display:none; }}
#ticks {{ position:relative; height:16px; }}
.tk {{ position:absolute; transform:translateX(-50%); top:2px; }}
#ctl {{ display:flex; gap:8px; align-items:center; margin-top:2px; }}
button {{ font:inherit; font-size:12px; padding:3px 10px; border-radius:6px;
          border:1px solid #c9c7bf; background:#fcfcfb; color:#1d1b17; cursor:pointer; }}
button:hover {{ border-color:#2a78d6; }}
#now {{ margin-left:auto; font-variant-numeric:tabular-nums; color:#1d1b17; }}
</style></head><body><div id="wrap">
<div id="bar" title="Click to play from here">{win_html}<div id="head"></div></div>
<div id="ticks">{tick_html}</div>
<div id="ctl"><button id="pa" title="Play the whole span">▶ span {t0:g}–{t1:g} s</button>
{win_btn}<button id="st" title="Stop">■ stop</button><span id="now"></span></div>
</div>
<audio id="a" preload="auto" src="{html.escape(src)}"></audio>
<script>
const C = {cfg};
const a = document.getElementById("a"), bar = document.getElementById("bar"),
      head = document.getElementById("head"), now = document.getElementById("now");
let stopAt = null;
function play(from, to) {{
  a.currentTime = Math.max(0, from - C.c0); stopAt = to; a.play();
}}
document.getElementById("pa").onclick = () => play(C.t0, C.t1);
const pw = document.getElementById("pw");
if (pw) pw.onclick = () => play(C.w[0], C.w[1]);
document.getElementById("st").onclick = () => {{ a.pause(); stopAt = null; }};
bar.onclick = (e) => {{
  const r = bar.getBoundingClientRect();
  play(C.t0 + (e.clientX - r.left) / r.width * (C.t1 - C.t0), C.t1);
}};
function frame() {{
  const t = C.c0 + a.currentTime;
  if (!a.paused || a.currentTime > 0) {{
    head.style.display = "block";
    head.style.left = (100 * (t - C.t0) / (C.t1 - C.t0)) + "%";
    now.textContent = t.toFixed(1) + " s";
  }}
  if (stopAt !== null && t >= stopAt) {{ a.pause(); stopAt = null; }}
  requestAnimationFrame(frame);
}}
frame();
</script></body></html>"""
