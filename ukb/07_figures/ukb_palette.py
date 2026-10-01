"""Shared palette + typography for the UKB epiBrainLLM figures (manuscript-v5).
Typeface follows the manuscript's existing Figures 1-5 (Arial; Liberation Sans is the metric-identical substitute
on machines without it). One import for every panel so colors/type never drift."""
import matplotlib as mpl

PAL = {
    # ---- semantic series (consistent across every figure) ----
    'enformer': '#1b6e5f',   # deep teal-green — epiBrainLLM, the hero series
    'enf_soft': '#d6e8e3',   # pale teal fill under the epiBrainLLM curve / win-region
    'snp':      '#9aa0a6',   # neutral grey    — raw SNP baseline
    'apoe':     '#2c6fbf',   # blue            — APOE reference locus
    'star':     '#e8a51c',   # warm gold       — the multi-locus combine
    'star_edge':'#8a6100',
    # ---- PRS baselines (one low-saturation set, never reused for the hero) ----
    'prscs':    '#3d6fa8',   # muted blue
    'ldpred2':  '#8c6bb1',   # muted purple
    'sdpr':     '#d98c3f',   # muted amber
    # ---- neutrals / semantic ----
    'alarm':    '#b2182b',
    'ink':      '#1f2328',
    'sub':      '#6a7079',
    'grid':     '#edeff2',
    'axis':     '#333940',
}

def _sans():
    """Arial as in the manuscript's Figures 1-5; Liberation Sans / Nimbus Sans are metric-identical substitutes
    on machines without Arial (e.g. the cluster). Drop Arial TTFs into ~/.fonts to get true Arial."""
    from matplotlib import font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in ['Arial', 'Liberation Sans', 'Arimo', 'Nimbus Sans', 'Helvetica'] if f in avail), 'DejaVu Sans')

def set_rc():
    sans = _sans()
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [sans, 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': sans, 'mathtext.it': sans + ':italic',
        'mathtext.bf': sans + ':bold', 'mathtext.cal': sans, 'mathtext.sf': sans, 'mathtext.tt': sans, 'mathtext.fallback': 'stixsans',
        'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.major.size': 2.8, 'ytick.major.size': 2.8,
        'axes.edgecolor': PAL['axis'], 'axes.grid': False,
        'axes.labelcolor': PAL['ink'], 'text.color': PAL['ink'],
        'xtick.color': PAL['axis'], 'ytick.color': PAL['axis'],
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 150,
    })
    return sans

MM = 1 / 25.4
