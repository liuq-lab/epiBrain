#!/usr/bin/env python3
"""Reformat Kunkle 2019 Stage-1 IGAP AD GWAS -> PRS-CS input format.

PRS-CS wants a whitespace/TAB file with header exactly:  SNP  A1  A2  BETA  SE
(A1 = effect allele, A2 = other allele). We keep only SNPs that also appear in
the LD-panel snpinfo (HapMap3) AND in the UKB validation bim, so PRS-CS doesn't
waste work on SNPs it will discard anyway. We also drop ambiguous/duplicate rows.

Kunkle columns:
  Chromosome Position MarkerName Effect_allele Non_Effect_allele Beta SE Pvalue
"""
import argparse, pandas as pd, numpy as np

def main(a):
    print("[read] Kunkle sumstats ...", flush=True)
    df = pd.read_csv(a.sumstats, sep=r'\s+')
    df = df.rename(columns={'MarkerName':'SNP','Effect_allele':'A1',
                            'Non_Effect_allele':'A2','Beta':'BETA','SE':'SE'})
    n0 = len(df)
    df['A1'] = df['A1'].str.upper(); df['A2'] = df['A2'].str.upper()
    df = df[['SNP','A1','A2','BETA','SE']].dropna()
    df = df[df['SE'] > 0]
    # restrict to LD-panel rsIDs (HapMap3) so PRS-CS uses the intended SNP set
    # NOTE: do NOT intersect with the LD panel here. Doing so silently drops
    # variants that are absent from HapMap3 -- including rs429358 (the APOE-e4
    # determinant, the single strongest AD variant, p=1e-881 in Kunkle). PRS-CS
    # itself decides which SNPs it can use (those in the LD panel); pre-filtering
    # only hides how many strong signals the panel misses. Keep the flag for
    # diagnostics but default to reporting, not filtering.
    if a.ld_snpinfo and a.filter_by_ld:
        ld = pd.read_csv(a.ld_snpinfo, sep=r'\s+')['SNP']
        df = df[df['SNP'].isin(set(ld))]
        print(f"[filter] kept {len(df)} of {n0} after LD-panel intersect", flush=True)
    elif a.ld_snpinfo:
        ld = set(pd.read_csv(a.ld_snpinfo, sep=r'\s+')['SNP'])
        n_in = df['SNP'].isin(ld).sum()
        print(f"[report] {n_in}/{len(df)} sumstats SNPs are in the LD panel "
              f"(the rest, e.g. rs429358, PRS-CS cannot weight)", flush=True)
    # restrict to UKB validation SNPs (rsID) as well
    if a.val_bim:
        vb = pd.read_csv(a.val_bim, sep=r'\s+', header=None, usecols=[1])[1]
        df = df[df['SNP'].isin(set(vb))]
        print(f"[filter] kept {len(df)} after UKB-bim intersect", flush=True)
    df = df.drop_duplicates('SNP')
    df.to_csv(a.out, sep='\t', index=False)
    print(f"[write] {len(df)} SNPs -> {a.out}", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sumstats', required=True)
    ap.add_argument('--ld_snpinfo', default='')
    ap.add_argument('--filter_by_ld', action='store_true',
                    help='(NOT recommended) drop sumstats SNPs absent from the LD panel')
    ap.add_argument('--val_bim', default='')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
