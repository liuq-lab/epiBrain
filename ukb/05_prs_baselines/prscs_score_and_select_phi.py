#!/usr/bin/env python3
"""Score UKB subjects with PRS-CS posterior weights and compute AD auROC.

Given a directory of PRS-CS per-chrom weight files (CHR SNP BP A1 A2 WEIGHT),
we run `plink2 --score` per chromosome (summing WEIGHT*dosage), add the
per-chrom sums into a genome-wide PRS per person, then evaluate auROC against
a phenotype column.

For a fair phi selection this is run once per phi grid value; the caller picks
the best-phi PRS. Because PRS-CS uses an EXTERNAL GWAS (Kunkle, no UKB overlap),
there is no train/test leakage: every UKB subject is a pure test subject, so we
score everyone with a phenotype and report a single auROC (no CV needed).
"""
import os as _os
EPIBRAIN_ROOT=_os.environ.get('EPIBRAIN_ROOT','.')
UKB_BFILE_DIR=_os.environ.get('UKB_BFILE_DIR','<path-to-UKB-plink-files>')
PLINK2=_os.environ.get('PLINK2','plink2')
UKB_COVAR_DIR=_os.environ.get('UKB_COVAR_DIR','<path-to-covariates>')
PYTHON=_os.environ.get('PYTHON','python3')
PYTHON=_os.environ.get('PYTHON','python3')
RSCRIPT=_os.environ.get('RSCRIPT','Rscript')
TABIX=_os.environ.get('TABIX','tabix')
import os, argparse, glob, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn import metrics

GENO = UKB_BFILE_DIR

def plink2_score(weight_file, chrom, keep_iids, plink2, tmp):
    """Return Series indexed by IID of score_sum for this chrom, or None."""
    cnum = str(chrom)
    # restrict plink2 to just the scored SNPs (huge speedup: avoids reading the
    # whole chromosome's variants for ~458k subjects)
    snplist = os.path.join(tmp, f'snps_chr{cnum}.txt')
    ws = pd.read_csv(weight_file, sep=r'\s+', header=None, usecols=[1])[1]
    ws.to_csv(snplist, index=False, header=False)
    # plink2 --score: cols = SNP(2) A1(4) WEIGHT(6) in the PRScs output
    out = os.path.join(tmp, f'sc_chr{cnum}')
    cmd = [plink2, '--bed', f'{GENO}/chr{cnum}_nodup.bed',
           '--bim', f'{GENO}/chr{cnum}_nodup.bim', '--fam', f'{GENO}/chr{cnum}_nodup.fam',
           '--extract', snplist,
           '--score', weight_file, '2', '4', '6', 'cols=+scoresums',
           '--out', out]
    if keep_iids:
        cmd += ['--keep', keep_iids]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sf = out + '.sscore'
    if not os.path.exists(sf):
        print(f"  [warn] chr{cnum} no sscore ({r.returncode}): {r.stderr.strip()[-200:]}")
        return None
    s = pd.read_csv(sf, sep=r'\s+')
    idc = 'IID' if 'IID' in s.columns else s.columns[0]
    # score-sum column is '<weight>_SUM' / 'SCORE1_SUM'; exclude NAMED_ALLELE_DOSAGE_SUM
    cands = [c for c in s.columns if c.upper().endswith('_SUM')
             and c.upper() != 'NAMED_ALLELE_DOSAGE_SUM']
    if not cands:
        print(f"  [warn] chr{cnum} no score-sum column: {list(s.columns)}"); return None
    sumcol = cands[-1]
    out_s = s.set_index(idc)[sumcol]; out_s.index = out_s.index.astype(str)
    return out_s

def main(a):
    wfiles = sorted(glob.glob(os.path.join(a.weight_dir, '*_chr*.txt')))
    if not wfiles:
        raise SystemExit(f"no weight files in {a.weight_dir}")
    # map chrom -> file
    chr2f = {}
    for f in wfiles:
        c = int(f.split('_chr')[-1].split('.')[0]); chr2f[c] = f
    print(f"[score] {len(chr2f)} chrom weight files from {a.weight_dir}")

    ad = pd.read_csv(a.ad_labels, sep='\t').set_index('eid'); ad.index = ad.index.astype(str)

    tmp = tempfile.mkdtemp()
    keep = None
    if a.keep:                       # optional subject restriction
        keep = a.keep

    total = None
    for c in sorted(chr2f):
        s = plink2_score(chr2f[c], c, keep, a.plink2, tmp)
        if s is None: continue
        total = s if total is None else total.add(s, fill_value=0.0)
        print(f"  chr{c}: +{len(s)} subj, running total {len(total)}", flush=True)

    prs = total.dropna()
    df = pd.DataFrame({'PRS': prs})
    df = df.join(ad[[a.label_col]], how='inner').dropna()
    df = df[df[a.label_col].isin([0, 1])]
    y = df[a.label_col].astype(int).values
    auc = metrics.roc_auc_score(y, df['PRS'].values)
    print(f"[AUC] {a.label_col}: PRS-CS auROC = {auc:.4f}  (n_case={int(y.sum())}, n_ctrl={int((y==0).sum())})")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write("model\tphi\tlabel\tauroc\tn_case\tn_ctrl\n")
        fh.write(f"PRScs_Kunkle\t{a.phi}\t{a.label_col}\t{auc:.4f}\t{int(y.sum())}\t{int((y==0).sum())}\n")
    # also persist the per-person PRS for reuse (e.g. covariate-adjusted models)
    if a.save_prs:
        df[['PRS', a.label_col]].to_csv(a.save_prs, sep='\t')
    print(f"-> {a.out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight_dir', required=True, help='dir with *_chr<N>.txt PRScs weights')
    ap.add_argument('--ad_labels', required=True)
    ap.add_argument('--label_col', default='ad_strict')
    ap.add_argument('--phi', default='NA')
    ap.add_argument('--keep', default='', help='optional plink keep file (FID IID)')
    ap.add_argument('--plink2', default='plink2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--save_prs', default='')
    main(ap.parse_args())
