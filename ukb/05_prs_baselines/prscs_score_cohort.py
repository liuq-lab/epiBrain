#!/usr/bin/env python3
"""Score genome-wide PRS-CS (phi1e-2 betas) for all UKB subjects -> scores_PRS-CS/chr*.sscore,
matching the LDpred2/SDPR score dirs, so the all-control benchmark can include PRS-CS consistently."""
import os as _os
EPIBRAIN_ROOT=_os.environ.get('EPIBRAIN_ROOT','.')
UKB_BFILE_DIR=_os.environ.get('UKB_BFILE_DIR','<path-to-UKB-plink-files>')
PLINK2=_os.environ.get('PLINK2','plink2')
UKB_COVAR_DIR=_os.environ.get('UKB_COVAR_DIR','<path-to-covariates>')
PYTHON=_os.environ.get('PYTHON','python3')
PYTHON=_os.environ.get('PYTHON','python3')
RSCRIPT=_os.environ.get('RSCRIPT','Rscript')
TABIX=_os.environ.get('TABIX','tabix')
import os,subprocess
BASE=EPIBRAIN_ROOT+'/UKB_section/01_benchmark_PRS'
G=UKB_BFILE_DIR; PLINK=PLINK2
betas=f'{BASE}/result/prscs_betas_all.txt'; OUT=f'{BASE}/scores_PRS-CS'; os.makedirs(OUT,exist_ok=True)
for c in range(1,23):
    o=f'{OUT}/chr{c}'
    if os.path.exists(f'{o}.sscore'): print(f'chr{c} exists, skip',flush=True); continue
    r=subprocess.run([PLINK,'--bfile',f'{G}/chr{c}_nodup','--score',betas,'1','2','3','cols=+scoresums','--keep',f'{BASE}/common_keep.txt',
                      '--out',o],capture_output=True,text=True)
    print(f'chr{c}: {"OK" if os.path.exists(f"{o}.sscore") else "FAIL "+r.stderr[-200:]}',flush=True)
print('DONE_SCORING',flush=True)
