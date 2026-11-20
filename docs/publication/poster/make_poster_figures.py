"""Poster copies of three paper figures, with larger in-figure text at poster scale.

Same data, same code paths as docs/publication/neurips/make_figures.py: each figure is drawn on a canvas
SCALE times the paper size, so when the poster scales it up to the same physical width its text is
1/SCALE times larger. Output: docs/publication/poster/figures_poster/ (poster.tex looks there first).

    python docs/publication/poster/make_poster_figures.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neurips"))
import matplotlib.pyplot as plt  # noqa: E402
import make_figures as mf  # noqa: E402

SCALE = 0.8
mf.OUT = HERE / "figures_poster"
_subplots = plt.subplots


def _scaled(*a, figsize=None, **k):
    if figsize is not None:
        figsize = (figsize[0] * SCALE, figsize[1] * SCALE)
    return _subplots(*a, figsize=figsize, **k)


plt.subplots = _scaled
mf.plt.subplots = _scaled

# The teaser's floor label "(item knowledge needed below)" overruns the smaller canvas and overstates
# Proposition 4 (item-varying predictions *can* score lower); the poster text explains the floor instead.
from matplotlib.axes import Axes  # noqa: E402

_text = Axes.text


def _text_short(self, x, y, s, *a, **k):
    if isinstance(s, str):
        s = s.replace(" (item knowledge needed below)", "")
    return _text(self, x, y, s, *a, **k)


Axes.text = _text_short
_xlabel = Axes.set_xlabel


def _xlabel_short(self, s, *a, **k):
    if isinstance(s, str):
        s = s.replace("local ALC, one subject per benchmark (lower is better)", "local ALC (lower is better)")
    return _xlabel(self, s, *a, **k)


Axes.set_xlabel = _xlabel_short
# per-figure scale: the heatmap's cell numbers crowd below 0.95; the rho plot's y label needs height.
for f, sc in ((mf.fig_teaser, 0.8), (mf.fig_rho_value, 0.85), (mf.fig_excess_map, 0.95)):
    SCALE = sc
    f()
