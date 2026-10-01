#!/bin/bash
# In-sample LD reference for LDpred2 and SDPR: 5,000 randomly chosen UK Biobank participants
# (ref/keep5k.txt, FID IID per line) extracted per chromosome from the QC'd imputed PLINK files.
UKB_BFILE_DIR=${UKB_BFILE_DIR:-<path-to-UKB-plink-files>}; PLINK2=${PLINK2:-plink2}
mkdir -p ref
for c in $(seq 1 22); do
  $PLINK2 --bfile $UKB_BFILE_DIR/chr${c}_nodup --keep ref/keep5k.txt --make-bed --out ref/ukb_chr${c}
done
