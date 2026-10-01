#!/bin/bash
# site-specific paths (set as environment variables before running)
EPIBRAIN_ROOT=${EPIBRAIN_ROOT:-$(pwd)}
# Genome-wide PRS-CS over a phi grid. Each (phi,chrom) is an independent job.
# Posterior SNP weights -> out/gw/phi<PHI>/ , one file per chrom.
set -u
BASE=${EPIBRAIN_ROOT}/z_work/prscs
PY=${EPIBRAIN_ROOT}/z_work/envs/tfenf/bin/python3
REF=$BASE/ldref/ldblk_1kg_eur
SST=$BASE/sumstats/kunkle_prscs.txt
NGWAS=63926
PHIS=${PHIS:-"1e-6 1e-4 1e-2 1"}
NITER=${NITER:-1000}
NBURN=${NBURN:-500}
MAXJOBS=${MAXJOBS:-12}

mkdir -p $BASE/out/gw $BASE/logs
run_one () {
  local phi=$1 chr=$2
  local tag=$(echo $phi | sed 's/[^0-9a-zA-Z]//g')
  local od=$BASE/out/gw/phi${tag}
  mkdir -p $od
  # skip if already done (any chr output present)
  if ls $od/gw_pst_eff_*_chr${chr}.txt >/dev/null 2>&1; then
    echo "[skip] phi=$phi chr=$chr exists"; return
  fi
  $PY $BASE/code/PRScs.py \
    --ref_dir=$REF \
    --bim_prefix=$BASE/valbim/perchr/ukb_chr${chr} \
    --sst_file=$SST --n_gwas=$NGWAS \
    --chrom=$chr --phi=$phi --n_iter=$NITER --n_burnin=$NBURN --seed=1 \
    --out_dir=$od/gw \
    > $BASE/logs/gw_phi${tag}_chr${chr}.log 2>&1
  echo "[done] phi=$phi chr=$chr rc=$?"
}
export -f run_one; export BASE PY REF SST NGWAS NITER NBURN

# build job list
JOBS=$BASE/logs/gw_jobs.txt; : > $JOBS
for phi in $PHIS; do for chr in $(seq 1 22); do echo "$phi $chr" >> $JOBS; done; done
echo "total jobs: $(wc -l < $JOBS), maxjobs=$MAXJOBS"

# run with a simple concurrency pool
xargs -a $JOBS -P $MAXJOBS -L1 bash -c 'run_one "$@"' _
echo "ALL_GW_DONE"
