#!/bin/bash
# site-specific paths (set as environment variables before running)
EPIBRAIN_ROOT=${EPIBRAIN_ROOT:-$(pwd)}
# Dense multi-worker extraction: pack multiple workers per GPU to hide the
# vcf2diploid CPU stage behind other workers' GPU inference. Each worker uses
# ~4.5GB VRAM (GPU has 80GB) so many fit. Worker skip-logic prevents duplicate
# subjects across shards.
#
# Usage: extract_dense.sh <GENE> <chrNN> <PHASED_VCF> <PER_GPU> <GPU_CSV>
#   e.g. extract_dense.sh APP chr21 z_work/geno_work/APP/APP_window_phased.vcf.gz 4 1,2,3,4,5,6,7
set -u
GENE=$1; CHRN=$2; PHASED=$3; PER_GPU=$4; GPUS=$5
PROJ=${EPIBRAIN_ROOT}
PY=$PROJ/z_work/envs/tfenf/bin/python
OUT=$PROJ/z_work/tf_feats_$GENE
mkdir -p "$OUT" "$PROJ/z_work/shards"
IFS=',' read -ra GPUARR <<< "$GPUS"
NG=${#GPUARR[@]}
NTOTAL=$((NG*PER_GPU))
echo "[dense] $GENE : $NG GPUs x $PER_GPU = $NTOTAL workers on GPUs [$GPUS]"
# split cohort into NTOTAL shards
rm -f "$PROJ/z_work/shards/${GENE}d_"* 2>/dev/null
split -n l/$NTOTAL -d -a 3 "$PROJ/z_work/cohort_all.txt" "$PROJ/z_work/shards/${GENE}d_"
export GENE_NAME=$GENE CHR_NAME=$CHRN PHASED_VCF=$PHASED
shard_idx=0
for g in "${GPUARR[@]}"; do
  for ((w=0; w<PER_GPU; w++)); do
    sh=$(printf "%s/z_work/shards/%sd_%03d" "$PROJ" "$GENE" "$shard_idx")
    CUDA_VISIBLE_DEVICES=$g $PY "$PROJ/z_work/tf_worker_gene.py" "$sh" "$OUT" \
        > "$PROJ/z_work/comparison_results/ex_${GENE}_g${g}_w${w}.log" 2>&1 &
    shard_idx=$((shard_idx+1))
    sleep 6
  done
done
echo "[dense] launched $shard_idx workers; waiting..."
wait
echo "[dense] $GENE ALL WORKERS RETURNED"
