#!/usr/bin/env python3
"""Build the UK Biobank figures of the epiBrainLLM manuscript.

    python make_figures.py                 # all three figures
    python make_figures.py fig6            # only Figure 6           (also: supp7, supp8)

Output (PNG at 450 dpi + vector PDF) is written to ./figures.  Canvases are fixed in millimetres, 178 mm wide.

    Figure 6                 figures/Fig6_UKB                   A-G   AD-centred main figure
    Supplementary Figure 7   figures/SuppFig7_dementia_forest   A-D   non-AD dementia + paired differences
    Supplementary Figure 8   figures/SuppFig8_MCI               A-D   MCI
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ukb_palette import set_rc, MM
import panels as P

OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
print('font used:', set_rc(), flush=True)

WIDTH = 178                                   # mm


def axes_mm(fig, fig_w, fig_h, x, y_top, w, h):
    """Add an axes whose position is given in millimetres (x from the left edge, y_top from the top edge)."""
    return fig.add_axes([x / fig_w, (fig_h - y_top - h) / fig_h, w / fig_w, h / fig_h])


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=450)
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)
    print('saved', OUT / name, flush=True)


def fig6(name='Fig6_UKB'):
    """Figure 6 (178 x 163 mm), panels in the order they are cited in the text:
         A  per-locus scatter, AD                 B  per-locus scatter, non-AD dementia     C  cumulative combine, AD
         D  single locus vs combine, AD (wide)                                              E  ROC, AD
         F  auROC bars, AD                        G  auROC bars, non-AD dementia"""
    H = 163
    P.SCALE = 0.82
    rc = {'font.size': 6.6, 'axes.labelsize': 6.8, 'xtick.labelsize': 5.9, 'ytick.labelsize': 5.9,
          'xtick.major.size': 2.0, 'ytick.major.size': 2.0}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(WIDTH * MM, H * MM))
        col = (13, 74, 135)          # left edges of the three columns (mm)
        S = 40                       # side of the square panels (mm)
        letter = dict(dx=-24, dy=2)

        def place(x, y_top, w, h, draw, cohort, tag, **kw):
            ax = axes_mm(fig, WIDTH, H, x, y_top, w, h)
            draw(ax, cohort, **kw)
            P.cohort_label(ax, cohort, dy=2)
            P.panel_letter(ax, tag, **letter)

        place(col[0], 6, S, S, P.scatter_panel, 'AD', 'A', legend=True)
        place(col[1], 6, S, S, P.scatter_panel, 'Dementia', 'B', legend=False)
        place(col[2], 6, S, S, P.curve_panel, 'AD', 'C', legend=True)
        place(13, 61, 101, S, P.combine_single_panel, 'AD', 'D')
        place(col[2], 61, S, S, P.roc_panel, 'AD', 'E')
        place(13, 120, 74, 32, P.bench_panel, 'AD', 'F')
        place(101, 120, 74, 32, P.bench_panel, 'Dementia', 'G')
        save(fig, name)
    P.SCALE = 1.0


def supp7(name='SuppFig7_dementia_forest'):
    """Supplementary Figure 7 (178 x 152 mm):
         A  cumulative combine, non-AD dementia   B  single locus vs combine, non-AD dementia
         C  ROC, non-AD dementia                  D  paired differences vs each baseline (AD + non-AD dementia)"""
    H = 152
    fig = plt.figure(figsize=(WIDTH * MM, H * MM))
    ax = axes_mm(fig, WIDTH, H, 14, 7, 58, 50)
    P.curve_panel(ax, 'Dementia'); P.cohort_label(ax, 'Dementia'); P.panel_letter(ax, 'A')
    ax = axes_mm(fig, WIDTH, H, 88, 7, 87, 50)
    P.combine_single_panel(ax, 'Dementia'); P.cohort_label(ax, 'Dementia'); P.panel_letter(ax, 'B')
    ax = axes_mm(fig, WIDTH, H, 14, 84, 56, 56)
    P.roc_panel(ax, 'Dementia'); P.cohort_label(ax, 'Dementia'); P.panel_letter(ax, 'C')
    ax = axes_mm(fig, WIDTH, H, 116, 84, 59, 56)
    P.forest_panel(ax, ['AD', 'Dementia']); P.panel_letter(ax, 'D', dx=-92)
    save(fig, name)


def supp8(name='SuppFig8_MCI'):
    """Supplementary Figure 8 (178 x 175 mm):  A  ROC   B  per-locus scatter   C  auROC bars   D  cumulative combine"""
    H = 175
    left, right, w = 14, WIDTH - 3 - 73, 73
    fig = plt.figure(figsize=(WIDTH * MM, H * MM))
    for x, y_top, h, draw, tag in [(left, 7, 73, P.roc_panel, 'A'), (right, 7, 73, P.scatter_panel, 'B'),
                                   (left, 97, 58, P.bench_panel, 'C'), (right, 97, 58, P.curve_panel, 'D')]:
        ax = axes_mm(fig, WIDTH, H, x, y_top, w, h)
        draw(ax, 'MCI')
        P.cohort_label(ax, 'MCI')
        P.panel_letter(ax, tag)
    save(fig, name)


if __name__ == '__main__':
    for target in (sys.argv[1:] or ['fig6', 'supp7', 'supp8']):
        {'fig6': fig6, 'supp7': supp7, 'supp8': supp8}[target]()
