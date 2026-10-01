#!/usr/bin/env python3
"""Per-locus (single-gene) auROC of epiBrainLLM vs raw SNP, computed from the SAME OOF matrices used by the benchmark
and the cumulative curves -> data/gene_rank_*.tsv (sorted by epiBrainLLM auROC). Single source for Fig 7A-D / Supp 8B."""
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
from sklearn.metrics import roc_auc_score
OOF=EPIBRAIN_ROOT+'/UKB_section/03_gene_selection'; DAT=EPIBRAIN_ROOT+'/UKB_section/FINAL_FIGURES/data'
for oof,out in [('oof_AD.npz','gene_rank_AD.tsv'),('oof_dementia.npz','gene_rank_dementia.tsv'),('oof_MCI_allctrl.npz','gene_rank_MCI_allctrl.tsv')]:
    z=np.load(f'{OOF}/{oof}',allow_pickle=True); y=z['y'].astype(int); genes=[str(g) for g in z['genes']]
    df=pd.DataFrame({'gene':genes,'indiv_Enf_auc':[roc_auc_score(y,z['enf'][:,i]) for i in range(len(genes))],
                     'indiv_SNP_auc':[roc_auc_score(y,z['snp'][:,i]) for i in range(len(genes))]}).sort_values('indiv_Enf_auc',ascending=False)
    df.to_csv(f'{DAT}/{out}',sep='\t',index=False)
    a=df.set_index('gene').loc['APOE']; print(f'{out:28s} n={len(y)}  epiBrainLLM>SNP at {int((df.indiv_Enf_auc>df.indiv_SNP_auc).sum())}/{len(df)} loci  APOE Enf={a.indiv_Enf_auc:.4f} SNP={a.indiv_SNP_auc:.4f}')
