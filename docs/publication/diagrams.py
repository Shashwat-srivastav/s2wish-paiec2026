"""Numeric-axis charts, strip plots, box-and-arrow diagrams and timelines for the research report.

Same conventions as charts.py: dependency-free SVG/HTML strings, theme tokens for every colour, a <title> tooltip
on every mark. Diagrams are drawn as inline SVG so the report is self-contained (no diagram library at view time).
"""
from __future__ import annotations

from charts import _fmt, _spread, _ticks, esc


def xy(series, *, xmin, xmax, ymin, ymax, xfmt=".1f", yfmt=".3f", xlabel="", ylabel="", w=720, h=320,
       label_w=150, refy=None, refy_label="", refx=None, refx_label="", xticks=None) -> str:
    """Numeric x axis. series: [{name, color, pts:[(x,y[,tip])], band?:[(x,lo,hi)], line?, markers?, dash?, r?, labels?}]."""
    L, R, T, B = 60, label_w, 16, 46
    pw, ph = w - L - R, h - T - B
    X = lambda v: L + pw * (v - xmin) / (xmax - xmin)
    Y = lambda v: T + ph * (1 - (min(max(v, ymin), ymax) - ymin) / (ymax - ymin))
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in _ticks(ymin, ymax):
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                 f'<text x="{L-8}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">{_fmt(t, yfmt)}</text>')
    for t in (xticks or _ticks(xmin, xmax, 6)):
        o.append(f'<text x="{X(t):.1f}" y="{T+ph+18}" class="tick" text-anchor="middle">{_fmt(t, xfmt)}</text>')
    if xlabel:
        o.append(f'<text x="{L+pw/2}" y="{h-6}" class="axis" text-anchor="middle">{esc(xlabel)}</text>')
    if ylabel:
        o.append(f'<text x="14" y="{T+ph/2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {T+ph/2})">{esc(ylabel)}</text>')
    if ymin < 0 < ymax:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    if refy is not None:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(refy):.1f}" y2="{Y(refy):.1f}" class="ref"/>'
                 f'<text x="{L+4}" y="{Y(refy)-5:.1f}" class="reflbl">{esc(refy_label)}</text>')
    if refx is not None:
        o.append(f'<line x1="{X(refx):.1f}" x2="{X(refx):.1f}" y1="{T}" y2="{T+ph}" class="ref"/>'
                 f'<text x="{X(refx)+4:.1f}" y="{T+10}" class="reflbl">{esc(refx_label)}</text>')
    ends = []
    for s in series:
        col = s["color"]
        if s.get("band"):
            up = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(hi):.1f}" for k, (x, lo, hi) in enumerate(s["band"]))
            dn = " ".join(f"L{X(x):.1f},{Y(lo):.1f}" for x, lo, hi in reversed(s["band"]))
            o.append(f'<path d="{up} {dn} Z" style="fill:{col}" opacity=".16" stroke="none"/>')
        pts = s["pts"]
        if s.get("line", True) and len(pts) > 1:
            dd = " ".join(f"{'M' if k == 0 else 'L'}{X(p[0]):.1f},{Y(p[1]):.1f}" for k, p in enumerate(pts))
            dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
            o.append(f'<path d="{dd}" style="stroke:{col}" class="ln" stroke-width="{s.get("width", 2)}"{dash}/>')
        if s.get("markers", True):
            for p in pts:
                tip = p[2] if len(p) > 2 else f'{s["name"]} · {_fmt(p[0], xfmt)}: {_fmt(p[1], yfmt)}'
                o.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{s.get("r", 4)}" class="dot" style="fill:{col}">'
                         f'<title>{esc(tip)}</title></circle>')
        for p in s.get("labels", []):  # [(x, y, text)] selective direct labels
            o.append(f'<text x="{X(p[0])+8:.1f}" y="{Y(p[1])+4:.1f}" class="vlbl">{esc(p[2])}</text>')
        if s.get("line", True) and s.get("endlabel", True) and pts:
            ends.append([Y(pts[-1][1]), s["name"], col])
    ys = _spread([e[0] for e in ends], 13)
    for (y0, name, col), y in zip(ends, ys):
        o.append(f'<line x1="{L+pw+4}" x2="{L+pw+12}" y1="{y:.1f}" y2="{y:.1f}" style="stroke:{col}" stroke-width="2"/>'
                 f'<text x="{L+pw+16}" y="{y+4:.1f}" class="dlbl">{esc(name)}</text>')
    o.append("</svg>")
    return "".join(o)


def strip(rows, *, vmin, vmax, fmt=".3f", w=720, row_h=34, label_w=210, xlabel="", ref=None, ref_label="") -> str:
    """One row per group, one dot per observation, a tick at the group mean. rows: [{label, values:[(v, tip)], color}]."""
    L, R, T, B = label_w, 96, 8, 40
    pw = w - L - R
    h = T + B + row_h * len(rows)
    X = lambda v: L + pw * (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in _ticks(vmin, vmax):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{T+row_h*len(rows)}" class="grid"/>'
                 f'<text x="{X(t):.1f}" y="{T+row_h*len(rows)+16}" class="tick" text-anchor="middle">{_fmt(t, fmt)}</text>')
    if ref is not None:
        o.append(f'<line x1="{X(ref):.1f}" x2="{X(ref):.1f}" y1="{T}" y2="{T+row_h*len(rows)}" class="ref"/>'
                 f'<text x="{X(ref)+4:.1f}" y="{T+10}" class="reflbl">{esc(ref_label)}</text>')
    for k, r in enumerate(rows):
        y = T + k * row_h + row_h / 2
        vals = [v for v, _ in r["values"]]
        o.append(f'<text x="{L-10}" y="{y+4:.1f}" class="rlbl" text-anchor="end">{esc(r["label"])}</text>'
                 f'<line x1="{L}" x2="{L+pw}" y1="{y:.1f}" y2="{y:.1f}" class="rowline"/>')
        for v, tip in r["values"]:
            o.append(f'<circle cx="{X(v):.1f}" cy="{y:.1f}" r="5.5" class="dot" style="fill:{r["color"]}" opacity=".85">'
                     f'<title>{esc(tip)}: {_fmt(v, fmt)}</title></circle>')
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            o.append(f'<line x1="{X(m):.1f}" x2="{X(m):.1f}" y1="{y-11:.1f}" y2="{y+11:.1f}" style="stroke:var(--fg)" stroke-width="2.5"/>'
                     f'<text x="{L+pw+8}" y="{y+4:.1f}" class="vlbl">mean {_fmt(m, fmt)}</text>')
        elif vals:
            o.append(f'<text x="{L+pw+8}" y="{y+4:.1f}" class="vlbl">{_fmt(vals[0], fmt)} (1 run)</text>')
    if xlabel:
        o.append(f'<text x="{L+pw/2}" y="{h-6}" class="axis" text-anchor="middle">{esc(xlabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def flow(nodes, edges, *, cw=200, rh=92, gx=30, gy=34, pad=10) -> str:
    """Box-and-arrow diagram on a grid. nodes: [{id, col, row, text, kind?, span?}] (text lines split on \\n, the
    first line is the title). edges: [(a, b)] or [(a, b, {label, dash})]. Arrows run between box edges."""
    ncol = max(n["col"] + n.get("span", 1) for n in nodes)
    nrow = max(n["row"] for n in nodes) + 1
    w, h = pad * 2 + ncol * cw - gx, pad * 2 + nrow * rh - gy
    box = {}
    for n in nodes:
        box[n["id"]] = (pad + n["col"] * cw, pad + n["row"] * rh, cw * n.get("span", 1) - gx, rh - gy)
    o = [f'<div class="scroll"><svg viewBox="0 0 {w} {h}" class="chart flow" role="img" style="min-width:{min(w, 620)}px">'
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 Z" class="arrh"/></marker></defs>']
    for e in edges:
        a, b = e[0], e[1]
        opt = e[2] if len(e) > 2 else {}
        ax, ay, aw, ah = box[a]
        bx, by, bw, bh = box[b]
        if opt.get("side") and abs(ay - by) >= 1:  # leave by the side, enter by the side
            right = bx > ax
            x1, x2 = (ax + aw, bx) if right else (ax, bx + bw)
            y1, y2 = ay + ah / 2, by + bh / 2
            mx = (x1 + x2) / 2
            path = f"M{x1:.1f},{y1:.1f} L{mx:.1f},{y1:.1f} L{mx:.1f},{y2:.1f} L{x2:.1f},{y2:.1f}"
            lx, ly = mx, (y1 + y2) / 2
        elif abs(ay - by) < 1:  # same row: side to side
            x1, x2 = (ax + aw, bx) if bx > ax else (ax, bx + bw)
            y1 = y2 = ay + ah / 2
            path = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"
            lx, ly = (x1 + x2) / 2, y1 - 5
        else:
            down = by > ay
            y1, y2 = (ay + ah, by) if down else (ay, by + bh)
            lo, hi = max(ax, bx) + 14, min(ax + aw, bx + bw) - 14
            if lo <= hi:  # boxes overlap horizontally: one straight vertical arrow
                x1 = x2 = (lo + hi) / 2
                path = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"
            else:  # elbow
                x1, x2 = ax + aw / 2, bx + bw / 2
                my = (y1 + y2) / 2
                path = f"M{x1:.1f},{y1:.1f} L{x1:.1f},{my:.1f} L{x2:.1f},{my:.1f} L{x2:.1f},{y2:.1f}"
            lx, ly = (x1 + x2) / 2 + 6, (y1 + y2) / 2 - 3
        dash = ' stroke-dasharray="5 4"' if opt.get("dash") else ""
        o.append(f'<path d="{path}" class="farrow"{dash} marker-end="url(#arr)"/>')
        if opt.get("label"):
            o.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="flbl" text-anchor="middle">{esc(opt["label"])}</text>')
    for n in nodes:
        x, y, bw, bh = box[n["id"]]
        lines = n["text"].split("\n")
        o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="7" class="fbox {n.get("kind", "")}"/>')
        y0 = y + bh / 2 - (len(lines) - 1) * 7.2 + 4
        for k, ln in enumerate(lines):
            o.append(f'<text x="{x+bw/2:.1f}" y="{y0+k*14.4:.1f}" text-anchor="middle" class="{"ft1" if k == 0 else "ft2"}">{esc(ln)}</text>')
    o.append("</svg></div>")
    return "".join(o)


def timeline(events) -> str:
    """Vertical timeline. events: [(date, title, detail_html, kind)] — kind in '', 'accent', 'good', 'bad', 'future'."""
    return '<ol class="tl">' + "".join(
        f'<li class="{k}"><span class="td">{esc(dt)}</span><span class="tb"><b>{esc(t)}</b><br>{body}</span></li>'
        for dt, t, body, k in events) + "</ol>"


EXTRA_CSS = r"""
.scroll{overflow-x:auto}
svg.flow .fbox{fill:var(--surface);stroke:var(--line);stroke-width:1.3}
svg.flow .fbox.accent{fill:var(--accent-soft);stroke:var(--accent)}
svg.flow .fbox.good{fill:color-mix(in oklab,var(--good) 12%,var(--surface));stroke:var(--good)}
svg.flow .fbox.bad{fill:color-mix(in oklab,var(--bad) 10%,var(--surface));stroke:var(--bad)}
svg.flow .fbox.warn{fill:color-mix(in oklab,var(--warn) 13%,var(--surface));stroke:var(--warn)}
svg.flow .fbox.muted{fill:var(--bg);stroke:var(--line);stroke-dasharray:4 3}
svg.flow .ft1{font:600 12.5px var(--body);fill:var(--fg)}
svg.flow .ft2{font:11.5px var(--body);fill:var(--fg2)}
svg.flow .flbl{font:11px var(--body);fill:var(--muted)}
svg.flow .farrow{fill:none;stroke:var(--fg2);stroke-width:1.4}
svg .arrh{fill:var(--fg2)}
ol.tl{list-style:none;margin:18px 0;padding:0;display:grid;gap:0}
ol.tl li{display:grid;grid-template-columns:100px minmax(0,1fr);gap:30px;max-width:none;position:relative;padding:0 0 18px}
ol.tl li:before{content:"";position:absolute;left:114px;top:6px;bottom:-6px;width:2px;background:var(--line)}
ol.tl li:last-child:before{display:none}
ol.tl .td{font:600 12px/1.6 var(--mono);color:var(--fg2);text-align:right}
ol.tl .tb{position:relative;padding-left:18px;font-size:13.5px;color:var(--fg2)}
ol.tl .tb b{font:600 15px/1.3 var(--display);color:var(--fg)}
ol.tl .tb:before{content:"";position:absolute;left:-24px;top:5px;width:12px;height:12px;border-radius:50%;background:var(--surface);border:2.5px solid var(--muted)}
ol.tl li.accent .tb:before{border-color:var(--accent);background:var(--accent)}
ol.tl li.good .tb:before{border-color:var(--good);background:var(--good)}
ol.tl li.bad .tb:before{border-color:var(--bad);background:var(--bad)}
ol.tl li.future .tb:before{border-style:dashed}
@media (max-width:560px){ol.tl li{grid-template-columns:minmax(0,1fr);gap:2px;padding-left:22px}ol.tl li:before{left:5px}ol.tl .td{text-align:left}ol.tl .tb{padding-left:0}ol.tl .tb:before{left:-22px}}
.abstract{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:18px 0}
.abstract p{max-width:none}
.kw{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0}
.kw span{font:12px/1.7 var(--mono);padding:0 9px;border-radius:999px;border:1px solid var(--line);color:var(--fg2);background:var(--bg)}
.cite{font:13px/1.6 var(--mono);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto;white-space:pre-wrap}
@media print{
  nav.toc,.dtools{display:none!important}
  .wrap{display:block;max-width:none}
  body{font-size:11.5px;padding-inline:0;background:#fff}
  h2{break-after:avoid;margin-top:34px} h3,.ft,.fs{break-after:avoid}
  figure,.card,.callout,svg,tr,.kpi{break-inside:avoid}
  details.dossier{break-inside:auto;border-radius:0}
  details.dossier>summary:before{content:""}
  .tablewrap,.scroll,pre.cmd,.cite{overflow:visible}
  table{font-size:10.5px} pre.cmd{white-space:pre-wrap;font-size:10px}
  a{color:inherit;text-decoration:none}
  @page{size:A4;margin:14mm 12mm}
}
"""
