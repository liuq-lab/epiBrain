#!/bin/bash
# site-specific paths (set as environment variables before running)
EPIBRAIN_ROOT=${EPIBRAIN_ROOT:-$(pwd)}
UKB_BFILE_DIR=${UKB_BFILE_DIR:-<path-to-UKB-plink-files>}
TABIX=${TABIX:-tabix}
# =============================================================================
# epiBrainLLM-UKB  Stage 1: genotype -> phased VCF -> personal diploid fasta
# =============================================================================
# Adapts AD-genomicLLM's WGS preprocessing to UKB imputed PLINK genotypes.
#
# Original (ADNI): vcf -> remove indels -> Beagle phase -> vcf2diploid
# UKB:  dedup PLINK bed (already SNV, hg19) -> per-gene-region subset
#       -> recode VCF -> Beagle phase -> bgzip/tabix -> vcf2diploid (per subject)
#
# KEY EFFICIENCY: Enformer only reads +/- ~200kb around each gene TSS, so we
# subset the genotypes to a window per gene BEFORE phasing. Phasing all 1.46M
# variants x 458k people is unnecessary; we phase only the ~thousands of
# variants in each gene window.
#
# Usage:
#   bash 01_build_personal_genomes.sh <GENE> <CHR> <TSS> <SUBJECT_LIST> [WIN_HALF]
# Example (APOE):
#   bash 01_build_personal_genomes.sh APOE 19 45409038 subjects_pilot.txt 250000
# =============================================================================
set -euo pipefail

GENE=$1            # e.g. APOE
CHR=$2             # e.g. 19  (no 'chr' prefix; matches .bim chrom col)
TSS=$3             # e.g. 45409038  (hg19 TSS from refGene_hg19_TSS.bed)
SUBJ=$4            # file: one UKB eid per line (subset of .fam IIDs)
WIN_HALF=${5:-320000}   # half-window in bp around TSS to keep (>= Enformer 196608/2 + slide)

PROJ=${EPIBRAIN_ROOT}
GENO_DIR=${UKB_BFILE_DIR}
REPO=$PROJ
REF=$PROJ/ukb/data/ref/chr${CHR}.fa
BEAGLE=$REPO/preprocess/beagle.22Jul22.46e.jar
MAP=$PROJ/ukb/data/genetic_maps/plink.chr${CHR}.GRCh37.map
VCF2DIP=$PROJ/ukb/data/tools/vcf2diploid/vcf2diploid.jar

WORK=$PROJ/z_work/geno_work/${GENE}
FASTA_OUT=$PROJ/z_work/fasta_mci/chr${CHR}
mkdir -p "$WORK" "$FASTA_OUT"

START=$(( TSS - WIN_HALF )); [ $START -lt 1 ] && START=1
END=$(( TSS + WIN_HALF ))

module load plink/2.00 2>/dev/null || true
module load bcftools 2>/dev/null || true

echo "[$(date +%H:%M:%S)] $GENE chr$CHR:$START-$END  (TSS=$TSS, win=$WIN_HALF)  n=$(wc -l < $SUBJ)"

# --- 1. subset PLINK to gene window + subjects, recode to VCF ---------------
# .fam IID == UKB eid; build keep file as FID IID (both = eid here).
awk '{print $1"\t"$1}' "$SUBJ" > "$WORK/keep.txt"
plink2 --bed   "$GENO_DIR/chr${CHR}_nodup.bed" \
       --bim   "$GENO_DIR/chr${CHR}_nodup.bim" \
       --fam   "$GENO_DIR/chr${CHR}_nodup.fam" \
       --chr   "$CHR" --from-bp "$START" --to-bp "$END" \
       --keep  "$WORK/keep.txt" \
       --export vcf bgz id-paste=iid \
       --out   "$WORK/${GENE}_window"
echo "[$(date +%H:%M:%S)] recoded VCF: $(bcftools index -n ${WORK}/${GENE}_window.vcf.gz 2>/dev/null || echo '?') variants"

# --- 2. Beagle phasing (reference-free), bundled jar ------------------------
java -Xmx16g -jar "$BEAGLE" \
     gt="$WORK/${GENE}_window.vcf.gz" \
     out="$WORK/${GENE}_window_phased" \
     map="$MAP" nthreads=8
${TABIX} -f -p vcf "$WORK/${GENE}_window_phased.vcf.gz"
echo "[$(date +%H:%M:%S)] phased + indexed. DONE $GENE (phased VCF ready; per-subject vcf2diploid is done by tf_worker at extraction time)."
# NOTE: the old per-subject vcf2diploid pre-generation loop was removed — the
# extraction worker (tf_worker_gene.py) runs vcf2diploid itself per subject, so
# pre-generating fastas here was redundant, slow, and littered vd_ temp dirs.
