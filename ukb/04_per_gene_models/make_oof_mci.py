#!/usr/bin/env python3
"""Rebuild per-gene OOF scores for MCI using ALL healthy controls (756 cases + ~4137 controls),
to shrink the AUC variance. Uses each gene's tuned config from all_genes_maxtune_ALL.tsv.
Enformer: winning PCA config + model, 10-fold OOF. SNP: cis-window genotypes, GBDT 10-fold OOF.
Output: UKB_section/03_gene_selection/oof_MCI_allctrl.npz (enf[N,23], snp[N,23], y, genes)."""
import os as _os
EPIBRAIN_ROOT=_os.environ.get('EPIBRAIN_ROOT','.')
UKB_BFILE_DIR=_os.environ.get('UKB_BFILE_DIR','<path-to-UKB-plink-files>')
PLINK2=_os.environ.get('PLINK2','plink2')
UKB_COVAR_DIR=_os.environ.get('UKB_COVAR_DIR','<path-to-covariates>')
PYTHON=_os.environ.get('PYTHON','python3')
PYTHON=_os.environ.get('PYTHON','python3')
RSCRIPT=_os.environ.get('RSCRIPT','Rscript')
TABIX=_os.environ.get('TABIX','tabix')
import os,glob,subprocess,tempfile,numpy as np,pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT='z_work'; OUT='UKB_section/03_gene_selection'
G=UKB_BFILE_DIR; PLINK=PLINK2
ad=pd.read_csv('ukb/phenotypes/ad_labels.tsv',sep='\t',usecols=['eid','ad_strict','ad_alldem','ad_mci'])
ad['eid']=ad['eid'].astype(str); ad=ad.set_index('eid')
g2l={}
for l in open('refGene_hg19_TSS.bed'):
    p=l.split('\t'); g2l[p[4]]=(p[0].replace('chr',''),int(p[1]))
cfg=pd.read_csv(f'{ROOT}/comparison_results/all_genes_maxtune_ALL.tsv',sep='\t')
cfg=cfg[cfg.disease=='MCI'].set_index('gene')
GENES=[str(g) for g in np.load(f'{OUT}/oof_MCI.npz',allow_pickle=True)['genes']]

def model_of(name,k):
    if name=='gb': m=GradientBoostingClassifier(n_estimators=150,learning_rate=0.07,max_depth=2,subsample=0.8,max_features='sqrt')
    else: m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C={'logC1':1.0,'logC.3':0.3,'logC3':3.0}[name]))
    return make_pipeline(SelectKBest(f_classif,k=k),m) if k else m
def mkX(pat,mat,mode,rows):
    if mode=='patmat': return np.concatenate([pat[rows],mat[rows]],1)
    if mode=='pat': return pat[rows]
    return (pat[rows]+mat[rows])/2
def oof(clf,X,y): return cross_val_predict(clf,X,y,cv=StratifiedKFold(10,shuffle=True,random_state=0),method='predict_proba',n_jobs=-1)[:,1]

# ---- common subjects + MCI cohort with ALL controls ----
dirs=[d for d in os.listdir(f'{ROOT}/pca_feats') if os.path.isdir(f'{ROOT}/pca_feats/{d}')]
common=None
for g in dirs:
    s=set(l.strip() for l in open(f'{ROOT}/pca_feats/{g}/subjects.txt')); common=s if common is None else common&s
case=sorted((set(ad[(ad.ad_mci==1)&(ad.ad_alldem==0)].index)&common),key=int)
ctrl=sorted(((set(ad[(ad.ad_alldem==0)&(ad.ad_mci==0)].index)&common)-set(case)),key=int)
subj=case+ctrl; y=np.array([1]*len(case)+[0]*len(ctrl)); N=len(subj)
print(f'MCI all-controls cohort: {len(case)} cases + {len(ctrl)} controls = {N}',flush=True)

ENF=np.zeros((N,len(GENES))); SNP=np.zeros((N,len(GENES)))
def snp_oof(gene):
    c,tss=g2l[gene]; tmp=tempfile.mkdtemp(dir=f'{ROOT}/tmp'); open(f'{tmp}/k','w').write('\n'.join(f'{x}\t{x}' for x in subj))
    subprocess.run([PLINK,'--bed',f'{G}/chr{c}_nodup.bed','--bim',f'{G}/chr{c}_nodup.bim','--fam',f'{G}/chr{c}_nodup.fam',
        '--chr',c,'--from-bp',str(tss-98304),'--to-bp',str(tss+98304),'--keep',f'{tmp}/k','--export','A','--out',f'{tmp}/g'],capture_output=True)
    if not os.path.exists(f'{tmp}/g.raw'): subprocess.run(['rm','-rf',tmp]); return np.full(N,0.5)
    raw=pd.read_csv(f'{tmp}/g.raw',sep=r'\s+'); raw['IID']=raw['IID'].astype(str); raw=raw.set_index('IID')
    cols=[c2 for c2 in raw.columns if c2 not in ('FID','PAT','MAT','SEX','PHENOTYPE')]
    Xs=raw.loc[subj,cols].values.astype(np.float32); cm=np.nan_to_num(np.nanmean(Xs,0)); ix=np.where(np.isnan(Xs)); Xs[ix]=np.take(cm,ix[1])
    subprocess.run(['rm','-rf',tmp]); return oof(GradientBoostingClassifier(n_estimators=100,max_depth=3),Xs,y)

for j,gene in enumerate(GENES):
    d=f'{ROOT}/pca_feats/{gene if gene!="APOE" else "apoe"}'
    ss=[l.strip() for l in open(f'{d}/subjects.txt')]; sidx={s:i for i,s in enumerate(ss)}; rows=[sidx[s] for s in subj]
    setting=cfg.loc[gene,'setting'] if gene in cfg.index else 'ad77_pca4/avg/logC.3/k20'  # paper default for genes absent from tune map
    v,mode,mname,ks=setting.split('/'); k=int(ks[1:])
    pat=np.load(f'{d}/{v}_pat.npy'); mat=np.load(f'{d}/{v}_mat.npy')
    X=mkX(pat,mat,mode,rows); k=min(k,X.shape[1])
    ENF[:,j]=oof(model_of(mname,k),X,y); del pat,mat,X
    SNP[:,j]=snp_oof(gene)
    print(f'  [{j+1:2d}/{len(GENES)}] {gene:9s} cfg={setting:30s} EnfAUC={roc_auc_score(y,ENF[:,j]):.4f} SnpAUC={roc_auc_score(y,SNP[:,j]):.4f}',flush=True)

np.savez(f'{OUT}/oof_MCI_allctrl.npz',enf=ENF,snp=SNP,y=y,genes=np.array(GENES))
print(f'>>> saved oof_MCI_allctrl.npz  N={N}',flush=True); print('DONE',flush=True)
