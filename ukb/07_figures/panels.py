#!/usr/bin/env python3
"""Panel library for the UK Biobank figures of the epiBrainLLM manuscript.

Every function draws ONE panel into a matplotlib Axes and reads only the small, aggregate files in ./data
(no individual-level records).  Panels carry no titles or footnotes: only a cohort label above the axes and a
panel letter; all descriptions live in the manuscript captions.

    scatter_panel         per-locus auROC, epiBrainLLM vs raw SNP        Fig 6A, 6B, Supp Fig 8B
    curve_panel           cumulative multi-locus combine (1..23 loci)    Fig 6C, Supp Fig 7A, Supp Fig 8D
    combine_single_panel  each single locus vs the multi-locus combine   Fig 6D, Supp Fig 7B
    roc_panel             ROC curves of the five methods                 Fig 6E, Supp Fig 7C, Supp Fig 8A
    bench_panel           auROC bars + 95% CI + significance             Fig 6F, 6G, Supp Fig 8C
    forest_panel          paired auROC differences vs each baseline      Supp Fig 7D
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from adjustText import adjust_text

from ukb_palette import PAL

DATA = Path(__file__).resolve().parent / 'data'
SCALE = 1.0          # global font/marker scale; make_figures.py lowers it for the compact Figure 6 layout

COHORT_LABEL = {'AD': 'AD', 'MCI': 'MCI', 'Dementia': 'Dementia (non-AD)'}
METHODS = ['epiBrainLLM', 'raw SNP', 'PRS-CS', 'LDpred2', 'SDPR']
COLOR = {'epiBrainLLM': PAL['enformer'], 'raw SNP': PAL['snp'],
         'PRS-CS': PAL['prscs'], 'LDpred2': PAL['ldpred2'], 'SDPR': PAL['sdpr']}
GENE_RANK = {'AD': 'gene_rank_AD.tsv', 'MCI': 'gene_rank_MCI_allctrl.tsv', 'Dementia': 'gene_rank_dementia.tsv'}
CURVE = {'AD': 'seed_curve_AD_full23.tsv', 'MCI': 'seed_curve_MCI_full23.tsv',
         'Dementia': 'seed_curve_dementia_full23.tsv'}

benchmark = pd.read_csv(DATA / 'benchmark3_source.csv')   # auROC + 95% CI per cohort x method
paired = pd.read_csv(DATA / 'paired_p3.csv')              # paired bootstrap: epiBrainLLM - baseline


def auc_of(cohort, method):
    """auROC of `method` in `cohort` (full precision, from benchmark3_source.csv)."""
    row = benchmark[(benchmark.disease == cohort) & (benchmark.method == method)]
    return float(row.auroc.iloc[0])


def cohort_label(ax, cohort, dy=5):
    ax.annotate(COHORT_LABEL[cohort], (0.5, 1), xycoords='axes fraction', xytext=(0, dy),
                textcoords='offset points', ha='center', va='bottom', fontsize=9.5 * SCALE,
                fontweight='normal', color=PAL['ink'], annotation_clip=False)


def panel_letter(ax, letter, dx=-30, dy=4):
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(dx, dy), textcoords='offset points',
                ha='left', va='bottom', fontsize=12 * SCALE, fontweight='normal', color=PAL['ink'],
                annotation_clip=False)


def _stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))


# ----------------------------------------------------------------------------------------------------------------
# ROC curves of the five methods
# ----------------------------------------------------------------------------------------------------------------
def roc_panel(ax, cohort, legend=True):
    """ROC curves.  data/roc_curve_<cohort>.tsv stores, per method, the cumulative numbers of false and true
    positives along the score threshold; dividing by the numbers of controls / cases gives FPR / TPR."""
    roc = pd.read_csv(DATA / f'roc_curve_{cohort}.tsv', sep='\t')
    size = benchmark[benchmark.disease == cohort].iloc[0]
    n_cases, n_controls = int(size.cases), int(size.controls)
    handles = []
    for method in reversed(METHODS):                      # epiBrainLLM is drawn last, i.e. on top
        r = roc[roc.method == method]
        fpr = r.false_positives.values / n_controls
        tpr = r.true_positives.values / n_cases
        auc = auc_of(cohort, method)
        assert abs(np.trapz(tpr, fpr) - auc) < 1e-9, f'ROC area != benchmark auROC ({cohort}, {method})'
        hero = (method == 'epiBrainLLM')
        lw = 1.9 if hero else 1.1
        ax.plot(fpr, tpr, color=COLOR[method], lw=lw, zorder=6 if hero else 3)
        handles.append(Line2D([0], [0], color=COLOR[method], lw=lw, label=f'{method} (AUC={auc:.3f})'))
    ax.plot([0, 1], [0, 1], ls='--', color='#1a237e', lw=0.8, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.grid(color=PAL['grid'], lw=0.6)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(handles=handles[::-1], loc='lower right', frameon=True, framealpha=0.95, edgecolor='#e0e0e0',
                  fontsize=6.6 * SCALE, handlelength=1.5, borderpad=0.5, labelspacing=0.4, handletextpad=0.5)


# ----------------------------------------------------------------------------------------------------------------
# auROC bars with 95% CI and significance versus epiBrainLLM
# ----------------------------------------------------------------------------------------------------------------
BENCH_YLIM = {'AD': (0.60, 0.735), 'MCI': (0.50, 0.60), 'Dementia': (0.58, 0.645)}   # y-axes are truncated


def bench_panel(ax, cohort):
    """Bars = auROC, error bars = 95% CI (Hanley-McNeil).  Asterisks inside a baseline's bar = one-sided paired
    bootstrap P for 'epiBrainLLM > this baseline' (* <0.05, ** <0.01, *** <0.001).  Brackets mark which methods
    need an external GWAS."""
    y0, y1 = BENCH_YLIM[cohort]
    d = benchmark[benchmark.disease == cohort].set_index('method')
    p = paired[paired.disease == cohort].set_index('baseline')
    for x, method in enumerate(METHODS):
        v, lo, hi = (float(d.loc[method, c]) for c in ('auroc', 'ci_lo', 'ci_hi'))
        hero = (method == 'epiBrainLLM')
        ax.bar(x, v - y0, bottom=y0, color=COLOR[method], width=0.74, edgecolor='white', lw=0.6, zorder=2)
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor='black', elinewidth=1.0,
                    capsize=2.8, capthick=0.9, zorder=4)
        ax.text(x, hi + (y1 - y0) * 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=7.4 * SCALE,
                fontweight='normal', color=COLOR['epiBrainLLM'] if hero else PAL['ink'])
        if not hero:
            stars = _stars(float(p.loc[method, 'p_onesided']))
            if stars:
                t = ax.text(x, lo - (y1 - y0) * 0.015, '$' + r'\ast' * len(stars) + '$', ha='center', va='top',
                            fontsize=9 * SCALE, color='white', zorder=5)
                t.set_path_effects([pe.withStroke(linewidth=1.4, foreground=PAL['ink'])])
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=7 * SCALE)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_ylabel('auROC')
    ax.grid(axis='y', color=PAL['grid'], lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    tr = ax.get_xaxis_transform()

    def bracket(x0, x1, label, col, y=-0.16):
        ax.plot([x0, x0, x1, x1], [y + 0.022, y, y, y + 0.022], transform=tr, color=col, lw=0.9, clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.015, label, transform=tr, ha='center', va='top', fontsize=6.8 * SCALE,
                color=col, fontweight='normal')

    bracket(-0.42, 1.42, 'no external GWAS', COLOR['epiBrainLLM'])
    bracket(1.58, 4.42, 'external GWAS', PAL['sub'])


# ----------------------------------------------------------------------------------------------------------------
# per-locus scatter: epiBrainLLM vs raw SNP
# ----------------------------------------------------------------------------------------------------------------
SCATTER_LIM = {'AD': (0.478, 0.715), 'MCI': (0.44, 0.585), 'Dementia': (0.44, 0.65)}


def scatter_panel(ax, cohort, legend=True):
    """One point per gene (all 23 labeled): x = raw-SNP auROC, y = epiBrainLLM auROC of the same 196,608-bp window.
    The star is the multi-locus combine (x = raw-SNP combine, y = epiBrainLLM combine)."""
    g = pd.read_csv(DATA / GENE_RANK[cohort], sep='\t')
    e = g.indiv_Enf_auc.values          # epiBrainLLM, single locus
    s = g.indiv_SNP_auc.values          # raw SNP, single locus
    lo, hi = SCATTER_LIM[cohort]
    win = e > s
    ax.plot([lo, hi], [lo, hi], color=PAL['sub'], lw=0.9, ls=(0, (5, 4)), zorder=1)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=PAL['enf_soft'], alpha=0.55, zorder=0)
    ax.scatter(s[win], e[win], s=9 * SCALE ** 2, color=PAL['enformer'], edgecolor='white', lw=0.3, zorder=3)
    ax.scatter(s[~win], e[~win], s=9 * SCALE ** 2, color=PAL['snp'], edgecolor='white', lw=0.3, zorder=3)
    texts = []
    for i, (xg, yg, gene) in enumerate(zip(s, e, g.gene)):
        col = PAL['apoe'] if gene == 'APOE' else (PAL['enformer'] if win[i] else PAL['sub'])
        texts.append(ax.text(xg, yg, gene, fontsize=4.3 * SCALE, color=col, ha='center', va='center'))
    for t in texts:
        t.set_path_effects([pe.withStroke(linewidth=1.2, foreground='white')])
        t.set_zorder(7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    wins = ax.text(0.035, 0.70 if legend else 0.965,
                   f'epiBrainLLM > raw SNP\nat {int(win.sum())} / {len(e)} loci', transform=ax.transAxes,
                   fontsize=6.6 * SCALE, va='top', ha='left', color=PAL['enformer'], linespacing=1.3)
    # adjustText nudges exactly-overlapping labels with np.random; fix the seed so the layout is reproducible
    rng_state = np.random.get_state()
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):          # silence adjustText's debug prints
        adjust_text(texts, x=s, y=e, ax=ax, objects=[wins], ensure_inside_axes=True, expand=(1.3, 1.7),
                    force_text=(0.6, 0.9), force_points=(0.3, 0.5), force_static=(0.5, 0.8), iter_lim=2000,
                    arrowprops=dict(arrowstyle='-', color='#b8bcc2', lw=0.35, shrinkA=1, shrinkB=1))
    np.random.set_state(rng_state)
    cs, ce = auc_of(cohort, 'raw SNP'), auc_of(cohort, 'epiBrainLLM')
    ax.scatter([cs], [ce], marker='*', s=160 * SCALE ** 2, color=PAL['star'], edgecolor=PAL['star_edge'],
               lw=0.8, zorder=6)
    ax.annotate('combine', (cs, ce), xytext=(cs - (hi - lo) * 0.035, ce), fontsize=6.6 * SCALE,
                color=PAL['star_edge'], ha='right', va='center', zorder=6)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.04))
    ax.set_xlabel('raw SNP auROC (per locus)')
    ax.set_ylabel('epiBrainLLM auROC (per locus)')
    if legend:
        h = [Line2D([0], [0], marker='o', ls='', mfc=PAL['enformer'], mec='white', ms=3.6, label='epiBrainLLM wins'),
             Line2D([0], [0], marker='o', ls='', mfc=PAL['snp'], mec='white', ms=3.6, label='raw SNP wins'),
             Line2D([0], [0], marker='*', ls='', mfc=PAL['star'], mec=PAL['star_edge'], ms=8,
                    label='multi-locus combine'),
             Line2D([0], [0], ls=(0, (5, 4)), color=PAL['sub'], lw=0.9, label='equal')]
        ax.legend(handles=h, loc='upper left', frameon=False, fontsize=6.2 * SCALE, handletextpad=0.4,
                  labelspacing=0.35, borderaxespad=0.3)


# ----------------------------------------------------------------------------------------------------------------
# each single locus vs the multi-locus combine
# ----------------------------------------------------------------------------------------------------------------
SINGLE_YAXIS = {'AD': (0.48, 0.72, 0.04), 'MCI': (0.44, 0.60, 0.04), 'Dementia': (0.48, 0.65, 0.04)}


def combine_single_panel(ax, cohort):
    """epiBrainLLM auROC of each of the 23 loci alone (APOE blue) and of the multi-locus combine (gold).
    Dotted line = chance (0.5); dashed line = APOE alone; the number is (combine - APOE) x 100."""
    genes = pd.read_csv(DATA / GENE_RANK[cohort], sep='\t')
    genes = genes.sort_values('indiv_Enf_auc', ascending=False).reset_index(drop=True)
    combine = auc_of(cohort, 'epiBrainLLM')
    apoe = float(genes[genes.gene == 'APOE'].indiv_Enf_auc.iloc[0])
    y0, y1, tick = SINGLE_YAXIS[cohort]
    xs = np.arange(len(genes))
    xc = len(genes) + 0.25
    colors = [PAL['apoe'] if g == 'APOE' else PAL['snp'] for g in genes.gene]
    ax.bar(xs, genes.indiv_Enf_auc - y0, bottom=y0, color=colors, width=0.74, edgecolor='white', lw=0.4, zorder=2)
    ax.bar([xc], [combine - y0], bottom=y0, color=PAL['star'], width=0.74, edgecolor='white', lw=0.4, zorder=2)
    ax.axhline(0.5, color=PAL['ink'], lw=0.8, ls=':', zorder=1)
    ax.axhline(apoe, color=PAL['apoe'], lw=1.0, ls='--', zorder=1)
    ax.annotate('', xy=(xc, combine), xytext=(xc, apoe),
                arrowprops=dict(arrowstyle='<->', color=PAL['star_edge'], lw=1.1), zorder=3)
    ax.text(xc, combine + 0.004, f'+{(combine - apoe) * 100:.1f}', color=PAL['star_edge'], fontsize=7.6 * SCALE,
            ha='center', va='bottom', fontweight='normal')
    ax.set_xticks(list(xs) + [xc])
    ax.set_xticklabels(list(genes.gene) + ['multi-locus\ncombine'], rotation=55, ha='right', fontsize=6.4 * SCALE)
    ax.set_xlim(-0.9, xc + 0.8)
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_locator(MultipleLocator(tick))
    ax.set_ylabel('auROC (single locus)')
    ax.grid(axis='y', color=PAL['grid'], lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------------------------------------------
# cumulative multi-locus combine
# ----------------------------------------------------------------------------------------------------------------
def curve_panel(ax, cohort, legend=True):
    """auROC of the combine as loci are added one at a time (ranked by single-locus epiBrainLLM auROC), for
    epiBrainLLM and raw SNP.  Points = mean over 6 random 10-fold splits, error bars = SD, band = gap."""
    d = pd.read_csv(DATA / CURVE[cohort], sep='\t')
    k = d.k.values
    ax.fill_between(k, d.SNP, d.Enf, color=PAL['enf_soft'], zorder=1)
    ax.errorbar(k, d.Enf, yerr=d.Enf_sd, fmt='-o', color=PAL['enformer'], ms=3.6, lw=1.8, mec='white', mew=0.5,
                ecolor='black', capsize=2, capthick=0.7, elinewidth=0.7, zorder=4)
    ax.errorbar(k, d.SNP, yerr=d.SNP_sd, fmt='-o', color=PAL['snp'], ms=3.6, lw=1.8, mec='white', mew=0.5,
                ecolor='#555555', capsize=2, capthick=0.7, elinewidth=0.7, zorder=3)
    lo = (d.SNP - d.SNP_sd).min()
    hi = (d.Enf + d.Enf_sd).max()
    rg = hi - lo
    ax.set_xticks([1, 5, 10, 15, 20, 23])
    ax.set_xlim(0.4, 23.6)
    ax.set_ylim(lo - 0.30 * rg, hi + 0.48 * rg)
    ax.set_xlabel('loci combined (cumulative)')
    ax.set_ylabel('auROC')
    ax.grid(axis='y', color=PAL['grid'], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    gap_first = d.gap.iloc[0] * 100
    gap_peak = d.gap.max() * 100
    k_peak = int(d.k.iloc[int(np.argmax(d.gap.values))])
    ax.text(0.03, 0.97,
            'Δ vs raw SNP (' + r'$\times10^{-2}$' + f')\n+{gap_first:.1f} (1 locus) → peak +{gap_peak:.1f} ({k_peak} loci)',
            transform=ax.transAxes, ha='left', va='top', fontsize=7 * SCALE, color=PAL['sub'], linespacing=1.25)
    if legend:
        h = [Line2D([0], [0], marker='o', color=PAL['enformer'], mfc=PAL['enformer'], mec='white', ms=5, lw=1.8,
                    label='epiBrainLLM'),
             Line2D([0], [0], marker='o', color=PAL['snp'], mfc=PAL['snp'], mec='white', ms=5, lw=1.8,
                    label='raw SNP')]
        ax.legend(handles=h, loc='lower right', frameon=False, fontsize=7.2 * SCALE, handlelength=2.2,
                  labelspacing=0.4)


# ----------------------------------------------------------------------------------------------------------------
# forest plot of paired differences
# ----------------------------------------------------------------------------------------------------------------
FOREST_BASELINES = ['SDPR', 'LDpred2', 'PRS-CS', 'raw SNP']       # bottom-to-top within a cohort


def forest_panel(ax, cohorts):
    """Paired difference epiBrainLLM - baseline (auROC x 100) with 95% paired-bootstrap CI; first cohort on top.
    Green + asterisk = CI excludes 0; grey = not significant."""
    green, grey = PAL['enformer'], '#9aa0a6'
    rows, headers, dividers = [], [], []
    y = 0.0
    order = list(reversed(cohorts))
    for gi, cohort in enumerate(order):
        sub = paired[paired.disease == cohort].set_index('baseline')
        ys = []
        for method in FOREST_BASELINES:
            r = sub.loc[method]
            sig = (r['sig'] == 'yes')
            rows.append((y, method, r['gap_x100'], r['gap_x100'] - r['ci_lo'], r['ci_hi'] - r['gap_x100'],
                         green if sig else grey, sig))
            ys.append(y)
            y += 1
        headers.append((max(ys) + 0.55, COHORT_LABEL[cohort]))
        if gi < len(order) - 1:
            dividers.append(y + 0.05)
        y += 1.1
    ax.axvline(0, color=PAL['ink'], lw=1.0, zorder=2)
    for yy, method, mean, err_lo, err_hi, col, sig in rows:
        ax.errorbar([mean], [yy], xerr=[[err_lo], [err_hi]], fmt='o', ms=6, color=col, mec='white', mew=0.8,
                    ecolor=col, elinewidth=1.8, capsize=4, capthick=1.5, zorder=4)
        if sig:
            ax.plot([mean + err_hi + 0.3], [yy], marker=(6, 2, 0), ms=6.5, mew=1.3, color=green, ls='', zorder=5)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([f'vs {r[1]}' for r in rows], fontsize=8.5 * SCALE)
    ax.tick_params(axis='y', length=0)
    for hy, label in headers:
        ax.text(-0.42, hy, label, transform=ax.get_yaxis_transform(), fontsize=10 * SCALE, fontweight='normal',
                ha='left', va='center', color=PAL['ink'])
    for dv in dividers:
        ax.axhline(dv, color='#e0e0e0', lw=0.8, zorder=0)
    ax.set_ylim(-0.7, max(r[0] for r in rows) + 0.9)
    ax.set_xlim(-3.0, 5.6)
    ax.set_xlabel(r'$\Delta$ auROC $\times\,10^{-2}$ (epiBrainLLM − baseline)', fontsize=8.5 * SCALE)
    ax.grid(axis='x', color='#eceff1', lw=0.7, zorder=0)
    ax.spines['left'].set_visible(False)
