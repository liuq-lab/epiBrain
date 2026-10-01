#!/bin/bash
# site-specific paths (set as environment variables before running)
EPIBRAIN_ROOT=${EPIBRAIN_ROOT:-$(pwd)}
PYTHON=${PYTHON:-python3}
PYTHON=${PYTHON:-python3}
RSCRIPT=${RSCRIPT:-Rscript}
cd ${EPIBRAIN_ROOT}/UKB_section/01_benchmark_PRS
export R_LIBS_USER=$PWD/Rlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=${RSCRIPT}
echo "=== LDpred2 start $(date) ===" > ldpred2_run.log
$R run_ldpred2.R >> ldpred2_run.log 2>&1
if [ -s result/ldpred2_betas_all.txt ]; then
  echo "=== scoring LDpred2 $(date) ===" >> ldpred2_run.log
  ${PYTHON} finalize_prs.py result/ldpred2_betas_all.txt LDpred2 >> ldpred2_run.log 2>&1
fi
echo "LDPRED2_ALL_DONE $(date)" >> ldpred2_run.log
