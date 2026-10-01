"""Generic per-gene Enformer feature worker. Reads gene/chr/VCF from env:
  GENE_NAME, CHR_NAME (e.g. chr2), PHASED_VCF.  Args: <shard_list> <outdir>
Output: <outdir>/<chr>_<eid>_{maternal,paternal}.npy = (3,896,5313)."""
import os as _os
EPIBRAIN_ROOT=_os.environ.get('EPIBRAIN_ROOT','.')
UKB_BFILE_DIR=_os.environ.get('UKB_BFILE_DIR','<path-to-UKB-plink-files>')
PLINK2=_os.environ.get('PLINK2','plink2')
UKB_COVAR_DIR=_os.environ.get('UKB_COVAR_DIR','<path-to-covariates>')
PYTHON=_os.environ.get('PYTHON','python3')
PYTHON=_os.environ.get('PYTHON','python3')
RSCRIPT=_os.environ.get('RSCRIPT','Rscript')
TABIX=_os.environ.get('TABIX','tabix')
import os, sys, math, time, subprocess, shutil, numpy as np
import tensorflow as tf
from pyfaidx import Fasta
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
PROJ=EPIBRAIN_ROOT
GENE=os.environ['GENE_NAME']; CHR=os.environ['CHR_NAME']
REF=f'{PROJ}/ukb/data/ref/{CHR}.fa'
PHASED=os.environ['PHASED_VCF']
VCF2DIP=f'{PROJ}/ukb/data/tools/vcf2diploid/vcf2diploid.jar'
JAVA=_os.environ.get('JAVA','java')
REFGENE=f'{PROJ}/AD-genomicLLM/refGene_hg19_TSS.bed'
TMPROOT=f'{PROJ}/z_work/tmp'
shard=sys.argv[1]; OUTDIR=sys.argv[2]; os.makedirs(OUTDIR, exist_ok=True)
model=tf.saved_model.load(f'{PROJ}/z_work/tf_enformer').model
print(f"[{GENE}] model loaded", flush=True)
gene2loc={l.split('\t')[4]:(l.split('\t')[0],int(l.split('\t')[1])) for l in open(REFGENE)}
chr_id,center=gene2loc[GENE]
start=center-100000; end=center+100000
SEQ_LENGTH=393216; interval=896*128
nb_regions=math.ceil((end-start-interval)/(2*interval))
def seq_to_mat(seq):
    d={'a':0,'A':0,'c':1,'C':1,'g':2,'G':2,'t':3,'T':3,'N':4,'n':4}
    mat=np.zeros((5,len(seq)))
    for i in range(len(seq)): mat[d[seq[i]],i]=1
    return mat[:4,:]
def feats_for(fa_path):
    key='maternal' if 'maternal' in fa_path else 'paternal'
    genome=Fasta(fa_path); out=[]
    for coor in range(center-interval*nb_regions, center+interval*(nb_regions+1), interval):
        seq=str(genome[f'{chr_id}_{key}'][(coor-SEQ_LENGTH//2):(coor+SEQ_LENGTH//2)])
        oh=np.expand_dims(seq_to_mat(seq).T,0).astype(np.float32)
        out.append(model.predict_on_batch(oh)['human'])
    return np.squeeze(np.stack(out))
subj=[l.strip() for l in open(shard) if l.strip()]
t0=time.time(); done=0
for eid in subj:
    mout=f'{OUTDIR}/{CHR}_{eid}_maternal.npy'; pout=f'{OUTDIR}/{CHR}_{eid}_paternal.npy'
    if os.path.exists(mout) and os.path.exists(pout): continue
    tmp=f'{TMPROOT}/vd_{GENE}_{eid}'; os.makedirs(tmp, exist_ok=True)
    try:
        subprocess.run([JAVA,'-Xmx8g','-jar',VCF2DIP,'-id',eid,'-chr',REF,'-vcf',PHASED],
                       cwd=tmp, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"[{GENE}] {eid} TIMEOUT", flush=True); shutil.rmtree(tmp,ignore_errors=True); continue
    ok=True
    for hap in ('maternal','paternal'):
        fa=f'{tmp}/{CHR}_{eid}_{hap}.fa'
        if not os.path.exists(fa): ok=False; break
        np.save(f'{OUTDIR}/{CHR}_{eid}_{hap}.npy', np.asarray(feats_for(fa)))
    shutil.rmtree(tmp, ignore_errors=True)
    if not ok: print(f"[{GENE}] {eid} FAILED", flush=True); continue
    done+=1
    if done%50==0: print(f"[{GENE}] {done} done {(time.time()-t0)/done:.1f}s/subj", flush=True)
print(f"[{GENE}] WORKER DONE: {done}, {time.time()-t0:.0f}s", flush=True)
