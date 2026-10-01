#!/usr/bin/env python3
"""Plan B (memory-safe): fit per-bin PCA on a 2000-subject sample, then transform
ALL subjects in batches (never loads everything at once -> no OOM). Stores compact
PCA features (pat & mat separate) for ad77 {4,8,12,16} + f5313 {8,12}; raw can then
be deleted. Usage: make_pca_planB.py <GENE> <CHR e.g. chr8>"""
import os,sys,glob,gc,numpy as np
from sklearn.decomposition import PCA
GENE=sys.argv[1]; CHR=sys.argv[2]
TFDIR=f'z_work/tf_feats_{GENE}'; OUT=f'z_work/pca_feats/{GENE}'; os.makedirs(OUT,exist_ok=True)
ad77=np.array([int(l.split('\t')[0]) for l in open('AD_contexts.txt')])
subj=sorted([os.path.basename(x).split('_')[1] for x in glob.glob(f'{TFDIR}/{CHR}_*_maternal.npy')
             if os.path.exists(x.replace('_maternal','_paternal'))], key=int)
open(f'{OUT}/subjects.txt','w').write('\n'.join(subj)+'\n')
N=len(subj); print(f"[{GENE}] {N} subjects",flush=True)
rng=np.random.RandomState(0); fit_idx=sorted(rng.choice(N,min(2000,N),replace=False))
fit_subj=[subj[i] for i in fit_idx]
CONFIGS=[('ad77',ad77,[4,8,12,16]),('f5313',None,[8,12])]
BATCH=3000
for hap in ('paternal','maternal'):
    th='pat' if hap=='paternal' else 'mat'
    # --- fit per-bin PCA on 2000-subject sample (load only those) ---
    print(f"[{GENE}] {hap}: fitting PCA on {len(fit_subj)} subj...",flush=True)
    Sfit=np.stack([np.load(f'{TFDIR}/{CHR}_{s}_{hap}.npy')[1] for s in fit_subj]).astype(np.float32) # (2000,896,5313)
    models={}  # (tag,k) -> list of 896 PCA
    for tag,tr,ks in CONFIGS:
        Sf=Sfit[:,:,tr] if tr is not None else Sfit
        for k in ks:
            models[(tag,k)]=[PCA(k,svd_solver='randomized',random_state=0).fit(Sf[:,j,:]) for j in range(896)]
            print(f"[{GENE}] fitted {tag}_pca{k}_{th}",flush=True)
    del Sfit,Sf; gc.collect()
    # --- transform ALL subjects in batches, accumulate ---
    acc={key:[] for key in models}
    for b0 in range(0,N,BATCH):
        bs=subj[b0:b0+BATCH]
        T=np.stack([np.load(f'{TFDIR}/{CHR}_{s}_{hap}.npy')[1] for s in bs]).astype(np.float32)
        for (tag,k),pcas in models.items():
            tr=ad77 if tag=='ad77' else None
            Tt=T[:,:,tr] if tr is not None else T
            acc[(tag,k)].append(np.concatenate([pcas[j].transform(Tt[:,j,:]) for j in range(896)],1).astype(np.float32))
        del T; gc.collect()
        print(f"[{GENE}] {hap} transformed {min(b0+BATCH,N)}/{N}",flush=True)
    for (tag,k),parts in acc.items():
        np.save(f'{OUT}/{tag}_pca{k}_{th}.npy', np.concatenate(parts,0))
    del acc,models; gc.collect()
sz=sum(os.path.getsize(f) for f in glob.glob(f'{OUT}/*.npy'))/1e9
print(f"[{GENE}] DONE. PCA feats={sz:.2f}GB",flush=True)
