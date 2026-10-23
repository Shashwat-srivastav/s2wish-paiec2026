"""Tiny dependency-free SVG chart kit for the research report.

Every chart returns an SVG (or HTML) string. Colours are theme tokens (CSS custom properties)
so charts follow the page's light/dark theme. Every mark carries a <title> tooltip, series are
direct-labelled, and each chart is paired with a data table by the caller.
"""
from __future__ import annotations

import html
import math

S = [f"var(--s{i})" for i in range(1, 9)]          # categorical slots, fixed order
GOOD, BAD, WARN = "var(--good)", "var(--bad)", "var(--warn)"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v: float, fmt: str) -> str:
    return format(v, fmt)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label y-positions apart so direct labels never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def line(series, xlabels, *, ymin, ymax, yfmt=".3f", ylabel="", xlabel="", h=300, w=720,
         ref=None, ref_label="", label_w=150) -> str:
    """series: [{name, ys, color, dash?, width?}] — x positions equally spaced."""
    L, R, T, B = 56, label_w, 16, 44
    pw, ph = w - L - R, h - T - B
    n = len(xlabels)
    X = lambda i: L + (pw * i / (n - 1) if n > 1 else pw / 2)
    Y = lambda v: T + ph * (1 - (v - ymin) / (ymax - ymin))
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in _ticks(ymin, ymax):
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                 f'<text x="{L-8}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">{_fmt(t, yfmt)}</text>')
    for i, xl in enumerate(xlabels):
        o.append(f'<text x="{X(i):.1f}" y="{T+ph+18}" class="tick" text-anchor="middle">{esc(xl)}</text>')
    if xlabel:
        o.append(f'<text x="{L+pw/2}" y="{h-6}" class="axis" text-anchor="middle">{esc(xlabel)}</text>')
    if ylabel:
        o.append(f'<text x="14" y="{T+ph/2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {T+ph/2})">{esc(ylabel)}</text>')
    if ref is not None:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" class="ref"/>'
                 f'<text x="{L+4}" y="{Y(ref)-5:.1f}" class="reflbl">{esc(ref_label)}</text>')
    ends = []
    for s in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(s["ys"]) if v is not None]
        d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        o.append(f'<path d="{d}" style="stroke:{s["color"]}" class="ln" stroke-width="{s.get("width", 2)}"{dash}/>')
        for i, v in enumerate(s["ys"]):
            if v is None:
                continue
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" class="dot" style="fill:{s["color"]}">'
                     f'<title>{esc(s["name"])} · {esc(xlabels[i])}: {_fmt(v, yfmt)}</title></circle>')
        last = [(i, v) for i, v in enumerate(s["ys"]) if v is not None][-1]
        ends.append([Y(last[1]), s["name"], s["color"], last[1]])
    ys = _spread([e[0] for e in ends], 13)
    for (y0, name, col, v), y in zip(ends, ys):
        o.append(f'<line x1="{L+pw+4}" x2="{L+pw+12}" y1="{y:.1f}" y2="{y:.1f}" style="stroke:{col}" stroke-width="2"/>'
                 f'<text x="{L+pw+16}" y="{y+4:.1f}" class="dlbl">{esc(name)} {_fmt(v, yfmt)}</text>')
    o.append("</svg>")
    return "".join(o)


def hbar(rows, *, vmin, vmax, fmt=".3f", ref=None, ref_label="", w=720, row_h=26, label_w=230,
         unit="") -> str:
    """rows: [{label, value, color, tip?, lo?, hi?}] — horizontal bars from 0 (or vmin)."""
    L, R, T, B = label_w, 70, 8, 28
    pw = w - L - R
    h = T + B + row_h * len(rows)
    X = lambda v: L + pw * (v - vmin) / (vmax - vmin)
    base = X(max(vmin, min(0, vmax)))
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in _ticks(vmin, vmax):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{T+row_h*len(rows)}" class="grid"/>'
                 f'<text x="{X(t):.1f}" y="{h-8}" class="tick" text-anchor="middle">{_fmt(t, fmt)}{unit}</text>')
    for k, r in enumerate(rows):
        y = T + k * row_h
        v = r["value"]
        x0, x1 = sorted((base, X(v)))
        bh = row_h - 10
        o.append(f'<text x="{L-10}" y="{y+row_h/2+4:.1f}" class="rlbl" text-anchor="end">{esc(r["label"])}</text>')
        o.append(f'<rect x="{x0:.1f}" y="{y+5}" width="{max(x1-x0, 1.5):.1f}" height="{bh}" rx="3" style="fill:{r["color"]}">'
                 f'<title>{esc(r.get("tip", r["label"]))}: {_fmt(v, fmt)}{unit}</title></rect>')
        if r.get("lo") is not None:
            o.append(f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y+row_h/2}" y2="{y+row_h/2}" class="ci"/>')
        tx = X(v) + (6 if v >= 0 or vmin >= 0 else -6)
        anchor = "start" if v >= 0 or vmin >= 0 else "end"
        o.append(f'<text x="{tx:.1f}" y="{y+row_h/2+4:.1f}" class="vlbl" text-anchor="{anchor}">{_fmt(v, fmt)}{unit}</text>')
    if vmin < 0 < vmax:
        o.append(f'<line x1="{base:.1f}" x2="{base:.1f}" y1="{T}" y2="{T+row_h*len(rows)}" class="zero"/>')
    if ref is not None:
        o.append(f'<line x1="{X(ref):.1f}" x2="{X(ref):.1f}" y1="{T}" y2="{T+row_h*len(rows)}" class="ref"/>'
                 f'<text x="{X(ref)+4:.1f}" y="{T+10}" class="reflbl">{esc(ref_label)}</text>')
    o.append("</svg>")
    return "".join(o)


def forest(rows, *, vmin, vmax, fmt=".1f", w=720, row_h=28, label_w=260, xlabel="",
           better="left") -> str:
    """rows: [{label, pt, lo, hi, color?}] — point estimate with CI, zero line."""
    L, R, T, B = label_w, 70, 8, 40
    pw = w - L - R
    h = T + B + row_h * len(rows)
    X = lambda v: L + pw * (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in _ticks(vmin, vmax):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{T+row_h*len(rows)}" class="grid"/>'
                 f'<text x="{X(t):.1f}" y="{T+row_h*len(rows)+16}" class="tick" text-anchor="middle">{_fmt(t, fmt)}</text>')
    if vmin < 0 < vmax:
        o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{T}" y2="{T+row_h*len(rows)}" class="zero"/>')
    for k, r in enumerate(rows):
        y = T + k * row_h + row_h / 2
        col = r.get("color") or (GOOD if r["hi"] < 0 and better == "left" else BAD if r["lo"] > 0 and better == "left" else "var(--muted)")
        o.append(f'<text x="{L-10}" y="{y+4:.1f}" class="rlbl" text-anchor="end">{esc(r["label"])}</text>'
                 f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" style="stroke:{col}" stroke-width="2.5" stroke-linecap="round"/>'
                 f'<circle cx="{X(r["pt"]):.1f}" cy="{y:.1f}" r="5.5" class="dot" style="fill:{col}">'
                 f'<title>{esc(r["label"])}: {_fmt(r["pt"], fmt)} [{_fmt(r["lo"], fmt)}, {_fmt(r["hi"], fmt)}]</title></circle>'
                 f'<text x="{X(max(r["hi"], r["pt"]))+8:.1f}" y="{y+4:.1f}" class="vlbl">{_fmt(r["pt"], fmt)}</text>')
    if xlabel:
        o.append(f'<text x="{L+pw/2}" y="{h-6}" class="axis" text-anchor="middle">{esc(xlabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def dots(categories, series, *, vmin, vmax, fmt=".2f", w=720, row_h=30, label_w=170, zero=True,
         xlabel="") -> str:
    """Cleveland dot plot: one row per category, one coloured dot per series (with optional CI)."""
    L, R, T, B = label_w, 30, 8, 40
    pw = w - L - R
    h = T + B + row_h * len(categories)
    X = lambda v: L + pw * (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in _ticks(vmin, vmax):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{T+row_h*len(categories)}" class="grid"/>'
                 f'<text x="{X(t):.1f}" y="{T+row_h*len(categories)+16}" class="tick" text-anchor="middle">{_fmt(t, fmt)}</text>')
    if zero and vmin < 0 < vmax:
        o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{T}" y2="{T+row_h*len(categories)}" class="zero"/>')
    ns = len(series)
    for k, c in enumerate(categories):
        y = T + k * row_h + row_h / 2
        o.append(f'<text x="{L-10}" y="{y+4:.1f}" class="rlbl" text-anchor="end">{esc(c)}</text>'
                 f'<line x1="{L}" x2="{L+pw}" y1="{y:.1f}" y2="{y:.1f}" class="rowline"/>')
        for j, s in enumerate(series):
            v = s["values"][k]
            if v is None:
                continue
            yy = y + (j - (ns - 1) / 2) * 5
            if s.get("lo"):
                o.append(f'<line x1="{X(s["lo"][k]):.1f}" x2="{X(s["hi"][k]):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" style="stroke:{s["color"]}" stroke-width="1.5" opacity=".55"/>')
            o.append(f'<circle cx="{X(v):.1f}" cy="{yy:.1f}" r="5" class="dot" style="fill:{s["color"]}">'
                     f'<title>{esc(s["name"])} · {esc(c)}: {_fmt(v, fmt)}</title></circle>')
    if xlabel:
        o.append(f'<text x="{L+pw/2}" y="{h-6}" class="axis" text-anchor="middle">{esc(xlabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def stacked(rows, parts, *, w=720, row_h=30, label_w=170, fmt=".0%") -> str:
    """100%-stacked horizontal bars. rows: [(label, [values per part])]; parts: [(name, color)]."""
    L, R, T, B = label_w, 20, 8, 12
    pw = w - L - R
    h = T + B + row_h * len(rows)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for k, (lab, vals) in enumerate(rows):
        y = T + k * row_h
        tot = sum(vals) or 1
        x = L
        o.append(f'<text x="{L-10}" y="{y+row_h/2+4:.1f}" class="rlbl" text-anchor="end">{esc(lab)}</text>')
        for (name, col), v in zip(parts, vals):
            wd = pw * v / tot
            if wd <= 0:
                continue
            o.append(f'<rect x="{x:.1f}" y="{y+4}" width="{max(wd-2, 0.5):.1f}" height="{row_h-8}" rx="2" style="fill:{col}">'
                     f'<title>{esc(lab)} · {esc(name)}: {v/tot:{fmt}} ({v:.2f})</title></rect>')
            if wd > 44:
                o.append(f'<text x="{x+wd/2:.1f}" y="{y+row_h/2+4:.1f}" class="inlbl" text-anchor="middle">{v/tot:{fmt}}</text>')
            x += wd
    o.append("</svg>")
    return "".join(o)


def hist(groups, *, bins=10, lo=0.0, hi=1.0, w=720, h=150, xlabel="", fmt=".1f") -> str:
    """Small-multiple histograms. groups: [(name, values, color)] laid out in a row."""
    n = len(groups)
    gw = w / n
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    for g, (name, vals, col) in enumerate(groups):
        counts = [0] * bins
        for v in vals:
            if v is None or v != v:
                continue
            k = min(int((v - lo) / (hi - lo) * bins), bins - 1)
            counts[max(k, 0)] += 1
        tot = max(sum(counts), 1)
        mx = max(counts) or 1
        x0, T, ph = g * gw + 10, 22, h - 50
        bw = (gw - 20) / bins
        o.append(f'<text x="{x0}" y="14" class="rlbl">{esc(name)} <tspan class="tick">n={sum(counts)}</tspan></text>')
        for i, c in enumerate(counts):
            bh = ph * c / mx
            o.append(f'<rect x="{x0+i*bw+1:.1f}" y="{T+ph-bh:.1f}" width="{bw-2:.1f}" height="{max(bh, 0.5):.1f}" rx="2" style="fill:{col}">'
                     f'<title>{esc(name)} {edges[i]:{fmt}}–{edges[i+1]:{fmt}}: {c} ({c/tot:.0%})</title></rect>')
        o.append(f'<line x1="{x0}" x2="{x0+gw-20}" y1="{T+ph}" y2="{T+ph}" class="zero"/>'
                 f'<text x="{x0}" y="{T+ph+14}" class="tick">{lo:{fmt}}</text>'
                 f'<text x="{x0+gw-20}" y="{T+ph+14}" class="tick" text-anchor="end">{hi:{fmt}}</text>')
    if xlabel:
        o.append(f'<text x="{w/2}" y="{h-4}" class="axis" text-anchor="middle">{esc(xlabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def heat(rows, cols, values, *, fmt=".3f", lo=None, hi=None, diverging=False, rev=False,
         caption="") -> str:
    """HTML heat table. Higher = darker (sequential) or signed (diverging around 0)."""
    flat = [v for r in values for v in r if v is not None]
    lo = min(flat) if lo is None else lo
    hi = max(flat) if hi is None else hi
    o = ['<div class="tablewrap"><table class="heat">']
    if caption:
        o.append(f"<caption>{esc(caption)}</caption>")
    o.append("<thead><tr><th></th>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr></thead><tbody>")
    for r, vals in zip(rows, values):
        o.append(f"<tr><th>{esc(r)}</th>")
        for v in vals:
            if v is None:
                o.append("<td>—</td>")
                continue
            if diverging:
                m = max(abs(lo), abs(hi)) or 1
                a = min(abs(v) / m, 1) * 70
                col = ("var(--divneg)" if (v < 0) != rev else "var(--divpos)")
            else:
                a = (0 if hi == lo else (v - lo) / (hi - lo)) * 70
                a = 70 - a if rev else a
                col = "var(--seq)"
            ink = "var(--onstrong)" if a > 45 else "var(--fg)"
            o.append(f'<td style="background:color-mix(in oklab, {col} {a:.0f}%, var(--surface));color:{ink}">{_fmt(v, fmt)}</td>')
        o.append("</tr>")
    o.append("</tbody></table></div>")
    return "".join(o)


def legend(items) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def table(header, rows, cls="data") -> str:
    o = [f'<div class="tablewrap"><table class="{cls}"><thead><tr>']
    o += [f"<th>{esc(h)}</th>" for h in header]
    o.append("</tr></thead><tbody>")
    for r in rows:
        o.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    o.append("</tbody></table></div>")
    return "".join(o)


def datatable(header, rows, title="Data table") -> str:
    return f'<details class="dt"><summary>{esc(title)}</summary>{table(header, rows)}</details>'
