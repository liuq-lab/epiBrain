#!/usr/bin/env python3
"""Cumulative L2-logistic combine curves over ALL 23 loci (marginal-AUC order, 6 random 10-fold splits) for AD / non-AD dementia / MCI
-> data/seed_curve_{AD,dementia,MCI}_full23.tsv. Same combiner as the benchmark (StandardScaler + LogisticRegression C=0.5)."""
import os as _os
EPIBRAIN_ROOT=_os.environ.get('EPIBRAIN_ROOT','.')
UKB_BFILE_DIR=_os.environ.get('UKB_BFILE_DIR','<path-to-UKB-plink-files>')
PLINK2=_os.environ.get('PLINK2','plink2')
UKB_COVAR_DIR=_os.environ.get('UKB_COVAR_DIR','<path-to-covariates>')
PYTHON=_os.environ.get('PYTHON','python3')
PYTHON=_os.environ.get('PYTHON','python3')
RSCRIPT=_os.environ.get('RSCRIPT','Rscript')
TABIX=_os.environ.get('TABIX','tabix')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
OOF=EPIBRAIN_ROOT+'/UKB_section/03_gene_selection'; DAT=EPIBRAIN_ROOT+'/UKB_section/FINAL_FIGURES/data'
def meta(): return make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=2000))
for oof,out in [('oof_AD.npz','seed_curve_AD_full23.tsv'),('oof_dementia.npz','seed_curve_dementia_full23.tsv'),('oof_MCI_allctrl.npz','seed_curve_MCI_full23.tsv')]:
    z=np.load(f'{OOF}/{oof}',allow_pickle=True); enf=z['enf'].astype(float); snp=z['snp'].astype(float); y=z['y'].astype(int); genes=[str(g) for g in z['genes']]
    order=[g for g,a in sorted([(g,roc_auc_score(y,enf[:,i])) for i,g in enumerate(genes)],key=lambda t:-t[1])]; gi={g:i for i,g in enumerate(genes)}
    def oofp(X,s): return X[:,0] if X.shape[1]==1 else cross_val_predict(meta(),X,y,cv=StratifiedKFold(10,shuffle=True,random_state=s),method='predict_proba',n_jobs=4)[:,1]
    rows=[]
    for k in range(1,len(order)+1):
        cols=[gi[g] for g in order[:k]]
        ea=np.array([roc_auc_score(y,oofp(enf[:,cols],s)) for s in range(6)]); sa=np.array([roc_auc_score(y,oofp(snp[:,cols],s)) for s in range(6)])
        rows.append(dict(k=k,gene=order[k-1],Enf=ea.mean(),Enf_sd=ea.std(),SNP=sa.mean(),SNP_sd=sa.std(),gap=(ea-sa).mean(),gap_sd=(ea-sa).std()))
        print(f'{out[11:14]} k={k:2d} {order[k-1]:9s} Enf={ea.mean():.4f} SNP={sa.mean():.4f} gap={(ea-sa).mean():+.4f}',flush=True)
    pd.DataFrame(rows).to_csv(f'{DAT}/{out}',sep='\t',index=False)
print('DONE',flush=True)
