#!/usr/bin/env python3
"""Score UKB genome-wide with a PRS beta file (SNP A1 beta), compute AD & MCI auROC
on the SAME balanced cohorts as the Enformer analysis (common gene-feature subjects).
Usage: finalize_prs.py <betas_file> <method_name>  [betas has no header: cols SNP A1 BETA]"""
import os as _os
EPIBRAIN_ROOT=_os.environ.get('EPIBRAIN_ROOT','.')
UKB_BFILE_DIR=_os.environ.get('UKB_BFILE_DIR','<path-to-UKB-plink-files>')
PLINK2=_os.environ.get('PLINK2','plink2')
UKB_COVAR_DIR=_os.environ.get('UKB_COVAR_DIR','<path-to-covariates>')
PYTHON=_os.environ.get('PYTHON','python3')
PYTHON=_os.environ.get('PYTHON','python3')
RSCRIPT=_os.environ.get('RSCRIPT','Rscript')
TABIX=_os.environ.get('TABIX','tabix')
import sys, subprocess, os, glob, numpy as np, pandas as pd
from sklearn import metrics
betas=sys.argv[1]; method=sys.argv[2]
ROOT=EPIBRAIN_ROOT+'/z_work'; BASE=EPIBRAIN_ROOT+'/UKB_section/01_benchmark_PRS'
G=UKB_BFILE_DIR; PLINK=PLINK2
OUT=f'{BASE}/scores_{method}'; os.makedirs(OUT,exist_ok=True)
# genome-wide PRS = sum of per-chr scoresums
prs=None
for c in range(1,23):
    o=f'{OUT}/chr{c}'
    subprocess.run([PLINK,'--bfile',f'{G}/chr{c}_nodup','--score',betas,'1','2','3','cols=+scoresums',
                    '--out',o],capture_output=True)
    sf=f'{o}.sscore'
    if not os.path.exists(sf): continue
    d=pd.read_csv(sf,sep=r'\s+');
    idc=[x for x in d.columns if x in ('IID','#IID')][0]; d[idc]=d[idc].astype(str)
    sumc=[x for x in d.columns if 'SCORE' in x.upper() and x.upper().endswith('SUM')][0]
    s=d.set_index(idc)[sumc]
    prs = s if prs is None else prs.add(s,fill_value=0)
prs.name='PRS'; prs.index=prs.index.astype(str)
print(f"[{method}] scored {len(prs)} subjects",flush=True)
# same common cohort as gene analysis
dirs=[os.path.basename(x.rstrip('/')) for x in glob.glob(f'{ROOT}/pca_feats/*/')]
def gsubj(g): return set(l.strip() for l in open(f'{ROOT}/pca_feats/{g}/subjects.txt'))
common=None
for g in dirs:
    s=gsubj(g); common=s if common is None else common&s
ad=pd.read_csv(f'{ROOT}/../ukb/phenotypes/ad_labels.tsv',sep='\t',usecols=['eid','ad_strict','ad_alldem','ad_mci'])
ad['eid']=ad['eid'].astype(str); ad=ad.set_index('eid')
healthy=set(ad[(ad.ad_alldem==0)&(ad.ad_mci==0)].index)
res={}
for dis,mask in [('AD',ad.ad_strict==1),('MCI',(ad.ad_mci==1)&(ad.ad_alldem==0))]:
    case=sorted([s for s in common if s in set(ad[mask].index) and s in prs.index],key=int)
    hp=[s for s in common if s in healthy and s not in set(case) and s in prs.index]
    rng=np.random.RandomState(0); ctrl=sorted(rng.choice(hp,size=min(len(case),len(hp)),replace=False).tolist(),key=int)
    subj=case+ctrl; y=np.array([1]*len(case)+[0]*len(ctrl)); x=prs.loc[subj].values
    auc=metrics.roc_auc_score(y,x); res[dis]=(len(subj),len(case),auc)
    print(f"[{method}] {dis}: n={len(subj)} ({len(case)}ca) auROC={auc:.4f}",flush=True)
with open(f'{BASE}/RESULT_{method}.txt','w') as f:
    for dis,(n,nc,a) in res.items(): f.write(f"{method}\t{dis}\t{n}\t{nc}\t{a:.4f}\n")
print(f"[{method}] DONE -> RESULT_{method}.txt",flush=True)
