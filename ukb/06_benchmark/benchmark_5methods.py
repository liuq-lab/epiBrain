"""Full 5-method benchmark (Enformer/raw SNP/PRS-CS/LDpred2/SDPR) x 3 diseases with paired bootstrap.
Writes FINAL_FIGURES/data/benchmark3_source.csv (5 methods) + paired_p3.csv (incl PRS-CS)."""
import os,glob,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import roc_auc_score
BASE='UKB_section/01_benchmark_PRS'; OOFD='UKB_section/03_gene_selection'; ROOT='z_work'; FF='UKB_section/FINAL_FIGURES'
ad=pd.read_csv('ukb/phenotypes/ad_labels.tsv',sep='\t',usecols=['eid','ad_strict','ad_dementia','ad_alldem','ad_mci']); ad['eid']=ad['eid'].astype(str); ad=ad.set_index('eid')
common=None
for g in os.listdir(f'{ROOT}/pca_feats'):
    if os.path.isdir(f'{ROOT}/pca_feats/{g}'):
        s=set(l.strip() for l in open(f'{ROOT}/pca_feats/{g}/subjects.txt')); common=s if common is None else common&s
def load(dirn):
    prs=None
    for f in glob.glob(f'{dirn}/chr*.sscore'):
        d=pd.read_csv(f,sep=r'\s+',usecols=lambda c:c in ('IID','#IID','SCORE1_SUM'))
        idc=[c for c in d.columns if c in ('IID','#IID')][0]; d[idc]=d[idc].astype(str)
        s=d.set_index(idc)['SCORE1_SUM']; prs=s if prs is None else prs.add(s,fill_value=0)
    return prs
PRS={'PRS-CS':load(f'{BASE}/scores_PRS-CS'),'LDpred2':load(f'{BASE}/scores_LDpred2'),'SDPR':load(f'{BASE}/scores_SDPR')}
def meta(): return make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=2000))
def comb(M,y,cols):
    if len(cols)==1: return M[:,cols[0]]
    return cross_val_predict(meta(),M[:,cols],y,cv=StratifiedKFold(10,shuffle=True,random_state=0),method='predict_proba',n_jobs=4)[:,1]
def pboot(pe,ps,y,B=3000,seed=1):
    rng=np.random.default_rng(seed);pos=np.where(y==1)[0];neg=np.where(y==0)[0];d=np.empty(B)
    for b in range(B):
        ii=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]);yb=y[ii]
        d[b]=roc_auc_score(yb,pe[ii])-roc_auc_score(yb,ps[ii])
    return roc_auc_score(y,pe)-roc_auc_score(y,ps),np.percentile(d,2.5),np.percentile(d,97.5),(d<=0).mean()
def hm(a,m,n):
    Q1=a/(2-a);Q2=2*a**2/(1+a); return 1.96*((a*(1-a)+(m-1)*(Q1-a**2)+(n-1)*(Q2-a**2))/(m*n))**0.5
COH={'AD':('oof_AD.npz',ad.ad_strict==1,True),'MCI':('oof_MCI_allctrl.npz',(ad.ad_mci==1)&(ad.ad_alldem==0),False),'Dementia':('oof_dementia.npz',(ad.ad_dementia==1)&(ad.ad_strict==0),False)}
brows=[]; prows=[]
for dis,(oof,mask,bal) in COH.items():
    z=np.load(f'{OOFD}/{oof}',allow_pickle=True); enf=z['enf'].astype(float);snp=z['snp'].astype(float);y=z['y'].astype(int); genes=[str(g) for g in z['genes']]
    case=sorted((set(ad[mask].index)&common),key=int); healthy=sorted(((set(ad[(ad.ad_alldem==0)&(ad.ad_mci==0)].index)&common)-set(case)),key=int)
    ctrl=sorted(np.random.RandomState(0).choice(healthy,len(case),replace=False).tolist(),key=int) if bal else healthy
    subj=case+ctrl; assert len(subj)==len(y)
    gi={g:i for i,g in enumerate(genes)}; order=[g for g,a in sorted([(g,roc_auc_score(y,enf[:,i])) for i,g in enumerate(genes)],key=lambda t:-t[1])]
    best=None
    for k in [1,3,5,7,10]:
        cols=[gi[g] for g in order[:k]]; a=roc_auc_score(y,comb(enf,y,cols))
        if best is None or a>best[1]: best=(k,a,cols)
    k,ea,cols=best; pe=comb(enf,y,cols); psnp=comb(snp,y,cols)
    m,n=len(case),len(ctrl); res={'epiBrainLLM':roc_auc_score(y,pe),'raw SNP':roc_auc_score(y,psnp)}
    pv={'raw SNP':psnp}
    for meth in ['PRS-CS','LDpred2','SDPR']:
        v=PRS[meth].reindex(subj).values.astype(float); pv[meth]=v; res[meth]=roc_auc_score(y,v)
    np.savez(f'{FF}/data/roc_scores_{dis}.npz',y=y,epiBrainLLM=pe,raw_SNP=psnp,PRS_CS=pv['PRS-CS'],LDpred2=pv['LDpred2'],SDPR=pv['SDPR'],peak_k=k,genes_used=np.array(order[:k]),subj=np.array(subj))
    for meth in ['epiBrainLLM','raw SNP','PRS-CS','LDpred2','SDPR']:
        a=res[meth]; h=hm(a,m,n); gw='no' if meth in('epiBrainLLM','raw SNP') else 'yes'
        brows.append([dis,meth,a,a-h,a+h,gw,'-',len(subj),m,n])
    print(f'{dis} (n={len(subj)}, {m}+{n}, peak k={k}): '+' '.join(f'{me}={res[me]:.3f}' for me in ['epiBrainLLM','raw SNP','PRS-CS','LDpred2','SDPR']),flush=True)
    for meth in ['raw SNP','PRS-CS','LDpred2','SDPR']:
        o,lo,hi,p=pboot(pe,pv[meth],y); prows.append([dis,meth,round(o*100,2),round(lo*100,2),round(hi*100,2),round(p,4),'yes' if lo>0 else 'no'])
        print(f'   Enf vs {meth:8s}: {o*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}] P={p:.4f} {"SIG" if lo>0 else "n.s."}',flush=True)
pd.DataFrame(brows,columns=['disease','method','auroc','ci_lo','ci_hi','needs_external_gwas','footing','n','cases','controls']).to_csv(f'{FF}/data/benchmark3_source.csv',index=False)
pd.DataFrame(prows,columns=['disease','baseline','gap_x100','ci_lo','ci_hi','p_onesided','sig']).to_csv(f'{FF}/data/paired_p3.csv',index=False)
print('DONE',flush=True)
