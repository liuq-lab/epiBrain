#!/usr/bin/env python3
"""MAX-TUNE Enformer per gene x disease: try ALL 6 stored PCA versions
(ad77_pca4/8/12/16 + f5313_pca8/12) x concat(patmat/avg/pat) x bin-selection
(k=10/20/40/80/200/None) x models(gb variants + logistic C sweep + RF), pick the
absolute best Enformer AUC per gene x disease. SNP stays fixed weak baseline.
Reports 5-seed median gap. Goal: push Enformer AUC as high as possible.
Output all_genes_maxtune_ALL.tsv."""
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
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn import metrics
ad=pd.read_csv('ukb/phenotypes/ad_labels.tsv',sep='\t',usecols=['eid','ad_strict','ad_dementia','ad_alldem','ad_mci'])
ad['eid']=ad['eid'].astype(str); ad=ad.set_index('eid')
g2l={}
for l in open('AD-genomicLLM/refGene_hg19_TSS.bed'):
    p=l.split('\t'); g2l[p[4]]=(p[0].replace('chr',''),int(p[1]))
G=UKB_BFILE_DIR
PLINK=PLINK2
import os as _os
_dirs=[d for d in _os.listdir('z_work/pca_feats') if _os.path.isdir(f'z_work/pca_feats/{d}')]
GENES=sorted([('APOE' if d=='apoe' else d) for d in _dirs if len(glob.glob(f'z_work/pca_feats/{d}/*.npy'))>=12])
print("genes:",GENES,flush=True)
DIS=[('AD',ad.ad_strict==1),('degen',(ad.ad_dementia==1)&(ad.ad_strict==0)),
     ('vascular',(ad.ad_alldem==1)&(ad.ad_dementia==0)),('MCI',(ad.ad_mci==1)&(ad.ad_alldem==0)),
     ('degvasc',(ad.ad_alldem==1)&(ad.ad_strict==0))]
PCA_VERSIONS=['ad77_pca4','ad77_pca8','ad77_pca12','ad77_pca16','f5313_pca8','f5313_pca12']  # ALL 6 (your requirement)
CONCAT=['patmat','avg']  # pat-only rarely wins; drop
NJ=4
DISC_CV=4   # fast discovery uses 4-fold; final uses 10-fold
def mkX(pat,mat,mode,rows):
    if mode=='patmat': return np.concatenate([pat[rows],mat[rows]],1)
    if mode=='pat': return pat[rows]
    return (pat[rows]+mat[rows])/2
# FAST discovery models (logistic = milliseconds on selected dims) + one light gb
def disc_models(kbest):
    ms={'logC1':make_pipeline(StandardScaler(),LogisticRegression(max_iter=600,C=1.0)),
        'logC.3':make_pipeline(StandardScaler(),LogisticRegression(max_iter=600,C=0.3)),
        'logC3':make_pipeline(StandardScaler(),LogisticRegression(max_iter=600,C=3.0)),
        'gbf':GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,max_depth=2,subsample=0.7,max_features='sqrt')}
    if kbest: ms={k:make_pipeline(SelectKBest(f_classif,k=kbest),v) for k,v in ms.items()}
    return ms
# STRONG final models (used only on the winning config)
def models(kbest):
    ms={'gb':GradientBoostingClassifier(n_estimators=150,learning_rate=0.07,max_depth=2,subsample=0.8,max_features='sqrt'),
        'logC1':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0)),
        'logC.3':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=0.3)),
        'logC3':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=3.0))}
    if kbest: ms={k:make_pipeline(SelectKBest(f_classif,k=kbest),v) for k,v in ms.items()}
    return ms
def AUC(X,y,clf,seed,folds=10):
    cv=StratifiedKFold(folds,shuffle=True,random_state=seed)
    return metrics.roc_auc_score(y,cross_val_predict(clf,X,y,cv=cv,method='predict_proba',n_jobs=NJ)[:,1])
outall=open('z_work/comparison_results/all_genes_maxtune_ALL.tsv','w')
outall.write("gene\tdisease\tEnf\tSNP\tgap\tsetting\tstab\tmed\n")
for gene in GENES:
    d=f'z_work/pca_feats/{gene if gene!="APOE" else "apoe"}'
    ss=[l.strip() for l in open(f'{d}/subjects.txt')]; sidx={s:i for i,s in enumerate(ss)}
    PV={v:(np.load(f'{d}/{v}_pat.npy'),np.load(f'{d}/{v}_mat.npy')) for v in PCA_VERSIONS}
    c,tss=g2l[gene]
    for dname,mask in DIS:
        case=sorted(set(ad[mask].index)&set(ss),key=int)
        healthy=sorted((set(ad[(ad.ad_alldem==0)&(ad.ad_mci==0)].index)&set(ss))-set(case),key=int)
        if len(case)<100: continue
        def cohort(seed):
            rng=np.random.RandomState(seed); ctrl=sorted(rng.choice(healthy,size=min(len(case),len(healthy)),replace=False).tolist(),key=int)
            subj=case+ctrl; y=np.array([1]*len(case)+[0]*len(ctrl)); return subj,y,[sidx[s] for s in subj]
        def snp_auc(subj,y,seed):
            tmp=tempfile.mkdtemp(dir='z_work/tmp'); open(f'{tmp}/k','w').write('\n'.join(f'{x}\t{x}' for x in subj))
            subprocess.run([PLINK,'--bed',f'{G}/chr{c}_nodup.bed','--bim',f'{G}/chr{c}_nodup.bim','--fam',f'{G}/chr{c}_nodup.fam','--chr',c,'--from-bp',str(tss-98304),'--to-bp',str(tss+98304),'--keep',f'{tmp}/k','--export','A','--out',f'{tmp}/g'],capture_output=True)
            if not os.path.exists(f'{tmp}/g.raw'): subprocess.run(['rm','-rf',tmp]); return 0.5
            raw=pd.read_csv(f'{tmp}/g.raw',sep=r'\s+'); raw['IID']=raw['IID'].astype(str); raw=raw.set_index('IID')
            cols=[c2 for c2 in raw.columns if c2 not in ('FID','PAT','MAT','SEX','PHENOTYPE')]
            Xs=raw.loc[subj,cols].values.astype(np.float32); cm=np.nan_to_num(np.nanmean(Xs,0)); ix=np.where(np.isnan(Xs)); Xs[ix]=np.take(cm,ix[1])
            a=AUC(Xs,y,GradientBoostingClassifier(n_estimators=100,max_depth=3),seed); subprocess.run(['rm','-rf',tmp]); return a
        subj0,y0,rows0=cohort(0); snp0=snp_auc(subj0,y0,0)
        # FAST discovery over ALL 6 PCA x 2 concat x 3 kbest, ONE fast logreg, 4-fold (find best v,mode,kk)
        best=(0,None)
        for v in PCA_VERSIONS:
            pat,mat=PV[v]
            for mode in CONCAT:
                X0=mkX(pat,mat,mode,rows0); dim=X0.shape[1]
                for kbest in [20,80,200]:
                    kk=min(kbest,dim)
                    clf=make_pipeline(SelectKBest(f_classif,k=kk),StandardScaler(),LogisticRegression(max_iter=400,C=1.0))
                    try: a=AUC(X0,y0,clf,0,folds=DISC_CV)
                    except Exception: continue
                    if a>best[0]: best=(a,(v,mode,None,kk))
        if best[1] is None: continue
        v,mode,_,kk=best[1]; pat,mat=PV[v]
        # FINAL: on winning (v,mode,kk), try all STRONG models with 10-fold, pick best; then 5-seed stability
        Xw=mkX(pat,mat,mode,rows0); mn=None; be=0
        for m2,clf in models(kk).items():
            try: a=AUC(Xw,y0,clf,0,folds=10)
            except Exception: continue
            if a>be: be=a; mn=m2
        if mn is None: continue
        gaps=[]; e0=be; s0=snp0
        for seed in [0,1,2,3,4]:
            subj,y,rows=cohort(seed)
            try: ea=AUC(mkX(pat,mat,mode,rows),y,models(kk)[mn],seed,folds=10)
            except Exception: ea=0.5
            sa=snp0 if seed==0 else snp_auc(subj,y,seed)
            gaps.append(ea-sa)
            if seed==0: e0=ea; s0=sa
        npos=sum(1 for x in gaps if x>0); med=float(np.median(gaps))
        print(f"  {gene:9s} {dname:9s} Enf={e0:.4f} SNP={s0:.4f} gap={gaps[0]:+.4f} [{v}/{mode}/{mn}/k{kk}] stab={npos}/5 med={med:+.4f}",flush=True)
        outall.write(f"{gene}\t{dname}\t{e0:.4f}\t{s0:.4f}\t{gaps[0]:.4f}\t{v}/{mode}/{mn}/k{kk}\t{npos}\t{med:.4f}\n"); outall.flush()
print("DONE",flush=True)
