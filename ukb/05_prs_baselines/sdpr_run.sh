#!/bin/bash
# SDPR (Zhou & Zhao 2021, https://github.com/eldronzhou/SDPR) with the in-sample UK Biobank LD reference.
# Summary statistics: sumstats/kunkle_sdpr.txt (columns SNP A1 A2 BETA P), N = 63,926 (Kunkle et al. 2019 stage 1).
# Step 1 builds the LD blocks from ref/ukb_chr<c> (see build_ukb_ld_reference.sh); step 2 runs the MCMC per chromosome.
SDPR=${SDPR:-./SDPR/SDPR}
mkdir -p ref result
for c in $(seq 1 22); do
  $SDPR -make_ref -ref_prefix ref/ukb_chr${c} -chr ${c} -ref_dir ref/
  $SDPR -mcmc -ref_dir ref/ -ss sumstats/kunkle_sdpr.txt -N 63926 -chr ${c} -out result/sdpr_chr${c}.txt
done
# concatenate (SNP A1 BETA) and score the cohort:
awk 'FNR>1{print $1,$2,$3}' result/sdpr_chr*.txt > result/sdpr_betas_all.txt
${PYTHON:-python3} score_prs_genomewide.py result/sdpr_betas_all.txt SDPR
