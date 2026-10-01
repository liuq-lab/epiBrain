# epiBrainLLM — UK Biobank validation

Code for the UK Biobank (UKB) validation of epiBrainLLM (manuscript Figure 6, Supplementary Figures 7–8,
Supplementary Tables 5–6). The ADNI analyses are described in the top-level README; this directory
re-implements the same genotype → personal genome → Enformer → prediction pipeline for UKB genotype
data and adds three genome-wide polygenic-score baselines.

UKB individual-level data cannot be redistributed. Everything under `07_figures/data/` is aggregate
(auROCs, confidence intervals, ROC curve counts) and is sufficient to regenerate every figure; all other
steps require access to UK Biobank (application 22783) and the external GWAS summary statistics.

## Pipeline

| Step | Directory | What it does | Manuscript |
|---|---|---|---|
| 1 | `01_cohort/` | AD / non-AD dementia / MCI case–control labels from a longitudinal phecode table of hospital (ICD-10, fields 41270/41280), death-register (40001/40002), primary-care (42040) and self-reported (20002) diagnoses. AD = phecode 290.11 (G30.x, F00.0–F00.2); non-AD dementia = 290.1 or 290.12 without 290.11; MCI = 292.2 (F06.7) without any dementia code; controls have no dementia (incl. vascular 290.16) and no MCI code. | Methods, *Study cohorts* |
| 2 | `02_personal_genomes/` | For each gene: `plink2` extracts the variants within ±320 kb of the TSS from the QC'd imputed PLINK files, Beagle 5.4 phases them (reference-free), `vcf2diploid` builds the maternal/paternal sequence of the 196,608-bp Enformer window on hg19. | Methods, *Construction of personal genomes* |
| 3 | `03_llm_features/` | `enformer_worker.py`: Enformer features for the three 114,688-bp regions around the TSS (3 × 896 bins × 5,313 tracks) per haplotype (shards packed on GPUs by `extract_dense.sh`). `make_pca_features.py`: per-bin PCA of the 77 brain-related tracks (`AD_contexts.txt`, 4–16 PCs) and of all 5,313 tracks (8/12 PCs), stored per haplotype. | Methods, *Genomic LLM features* |
| 4 | `04_per_gene_models/` | Per-gene prediction scores. `select_per_gene_config.py` chooses, for each gene and cohort, the epiBrainLLM feature representation (PCA version, haplotype concatenation vs. average, number of selected bins) and base learner (gradient boosting or L2-logistic regression) by cross-validated auROC; `make_oof_*.py` then produce 10-fold out-of-fold scores with the selected configuration (AD / non-AD dementia / MCI). The raw-SNP baseline uses all variants of the same window with a gradient-boosting classifier. | Methods, *epiBrainLLM features and multi-locus combine* |
| 5 | `05_prs_baselines/` | PRS-CS (1000 Genomes EUR LD, φ grid), LDpred2-auto and SDPR trained on Kunkle et al. 2019 stage-1 summary statistics (N = 63,926) and scored on the UKB cohorts with `plink2 --score`. LDpred2/SDPR use an in-sample LD reference of 5,000 UKB participants (`build_ukb_ld_reference.sh`). | Methods, *Polygenic-score baselines* |
| 6 | `06_benchmark/` | `benchmark_5methods.py`: L2-logistic stacking of the per-gene scores (loci ranked by marginal auROC, k ∈ {1,3,5,7,10} chosen by cross-validated auROC), Hanley–McNeil CIs, stratified paired bootstrap vs. each baseline (3,000 resamples), per-subject ROC scores. `make_gene_rank.py`, `make_cumulative_curves.py`: per-locus auROCs and cumulative-combine curves over all 23 loci (6 random 10-fold splits). | Methods, *Statistical evaluation*; Supp. Table 6 |
| 7 | `07_figures/` | Self-contained figure code + aggregate data (see its README). | Figure 6, Supp. Fig. 7–8 |

Gene panel: the 20 ADSP Gene Verification Committee AD genes plus ACE, HLA-DRB1 and PTK2B
(Supplementary Table 5); TSS coordinates from `../refGene_hg19_TSS.bed`.

## Running

Scripts read site-specific locations from environment variables (defaults are placeholders):

```
EPIBRAIN_ROOT   project root (scripts use paths relative to it)
UKB_BFILE_DIR   QC'd UKB imputed genotypes, PLINK format, one file set per chromosome (chr<N>_nodup.*)
UKB_COVAR_DIR   covariate table (age, sex, genetic PCs)
PLINK2, TABIX, JAVA, PYTHON, RSCRIPT   executables
```

Steps 2–4 are run per gene (23 genes × 13,574 participants); step 3 needs GPUs (Enformer,
TensorFlow 2.15, `tfhub` Enformer model as in the top-level README). Step 5 needs
[PRS-CS](https://github.com/getian107/PRScs), [bigsnpr](https://privefl.github.io/bigsnpr/) (LDpred2) and
[SDPR](https://github.com/eldronzhou/SDPR); their installation is not bundled here.
Step 7 only needs Python ≥ 3.10 with numpy, pandas, matplotlib and adjustText (`07_figures/requirements.txt`):

```bash
cd ukb/07_figures && python make_figures.py
```

## Software versions used

Python 3.12 (scikit-learn 1.7.2, numpy 1.26, pandas 2.3, matplotlib 3.10), TensorFlow 2.15.1 (Enformer feature extraction),
PLINK 2.0 (a.6.13), Beagle 5.4 (22Jul22.46e), vcf2diploid v0.2.6a, PRS-CS (2024), bigsnpr 1.12 (LDpred2), SDPR 0.9.1, R 4.4.0.
