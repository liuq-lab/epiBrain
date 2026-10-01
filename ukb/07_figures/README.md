# UK Biobank figures of the epiBrainLLM manuscript — plotting code

Everything needed to redraw **Figure 6**, **Supplementary Figure 7** and **Supplementary Figure 8**.
The `data/` folder holds only small aggregate tables (auROCs, confidence intervals, ROC-curve counts);
it contains **no individual-level UK Biobank records and no participant IDs**.

```
pip install -r requirements.txt
python make_figures.py            # writes figures/*.png (450 dpi) and figures/*.pdf
python make_figures.py fig6       # one figure only: fig6 | supp7 | supp8
```

The output is deterministic: two runs give byte-identical PNG files (the gene-label layout uses a fixed random seed).
Fonts: Arial if installed, otherwise Liberation Sans (metric-identical), chosen automatically in `ukb_palette.py`.

## Which code draws which panel

| Manuscript | Panel | Content | Function in `panels.py` | Data files in `data/` |
|---|---|---|---|---|
| **Figure 6** (`fig6()`) | A | per-locus auROC, epiBrainLLM vs raw SNP, AD | `scatter_panel('AD')` | `gene_rank_AD.tsv`, `benchmark3_source.csv` (star) |
| | B | same, non-AD dementia | `scatter_panel('Dementia')` | `gene_rank_dementia.tsv`, `benchmark3_source.csv` |
| | C | cumulative multi-locus combine, AD | `curve_panel('AD')` | `seed_curve_AD_full23.tsv` |
| | D | each single locus vs the combine, AD | `combine_single_panel('AD')` | `gene_rank_AD.tsv`, `benchmark3_source.csv` |
| | E | ROC curves of the five methods, AD | `roc_panel('AD')` | `roc_curve_AD.tsv`, `benchmark3_source.csv` |
| | F | auROC bars + 95% CI + significance, AD | `bench_panel('AD')` | `benchmark3_source.csv`, `paired_p3.csv` |
| | G | same, non-AD dementia | `bench_panel('Dementia')` | `benchmark3_source.csv`, `paired_p3.csv` |
| **Supplementary Figure 7** (`supp7()`) | A | cumulative combine, non-AD dementia | `curve_panel('Dementia')` | `seed_curve_dementia_full23.tsv` |
| | B | single locus vs combine, non-AD dementia | `combine_single_panel('Dementia')` | `gene_rank_dementia.tsv`, `benchmark3_source.csv` |
| | C | ROC curves, non-AD dementia | `roc_panel('Dementia')` | `roc_curve_Dementia.tsv`, `benchmark3_source.csv` |
| | D | paired differences vs each baseline (AD, non-AD dementia) | `forest_panel(['AD','Dementia'])` | `paired_p3.csv` |
| **Supplementary Figure 8** (`supp8()`) | A | ROC curves, MCI | `roc_panel('MCI')` | `roc_curve_MCI.tsv`, `benchmark3_source.csv` |
| | B | per-locus auROC, MCI | `scatter_panel('MCI')` | `gene_rank_MCI_allctrl.tsv`, `benchmark3_source.csv` |
| | C | auROC bars, MCI | `bench_panel('MCI')` | `benchmark3_source.csv`, `paired_p3.csv` |
| | D | cumulative combine, MCI | `curve_panel('MCI')` | `seed_curve_MCI_full23.tsv` |

Supplementary Table 6 of the manuscript is `benchmark3_source.csv` joined with `paired_p3.csv`.

## Files

| File | Purpose |
|---|---|
| `make_figures.py` | entry point; composes panels on fixed millimetre canvases (178 mm wide) |
| `panels.py` | one function per panel type (table above) |
| `ukb_palette.py` | colours and matplotlib rc (font, line widths) |
| `figures/` | the rendered figures. Supplementary Figures 7–8 are byte-identical to those in the manuscript; Figure 6 differs only in the automatic placement of the gene labels in panel A (the manuscript version was drawn before the label layout was given a fixed seed) |
| `upstream_analysis/` | scripts that produced `data/` from individual-level data (reference only, see below) |

## Data dictionary

| File | Columns |
|---|---|
| `benchmark3_source.csv` | `disease` (AD, MCI, Dementia = non-AD dementia), `method`, `auroc`, `ci_lo`, `ci_hi` (95% Hanley–McNeil), `needs_external_gwas`, `n`, `cases`, `controls` |
| `paired_p3.csv` | `disease`, `baseline`, `gap_x100` = (auROC epiBrainLLM − auROC baseline) × 100, `ci_lo`, `ci_hi` (95% stratified paired bootstrap, 3,000 resamples, × 100), `p_onesided`, `sig` (CI excludes 0) |
| `gene_rank_*.tsv` | `gene`, `indiv_Enf_auc` = single-locus auROC of epiBrainLLM, `indiv_SNP_auc` = single-locus auROC of raw SNP |
| `seed_curve_*_full23.tsv` | `k` loci combined, `gene` added at step k, `Enf`/`SNP` = mean auROC over 6 random 10-fold splits, `Enf_sd`/`SNP_sd`, `gap` = Enf − SNP, `gap_sd` |
| `roc_curve_*.tsv` | `method`, `false_positives`, `true_positives` = cumulative counts along the score threshold; FPR = false_positives / controls, TPR = true_positives / cases |

In file and column names, `Enf` means epiBrainLLM (Enformer-derived features) and `SNP` means the raw-genotype baseline.

## Cohorts

| Cohort | Cases | Controls | n |
|---|---|---|---|
| AD | 4,124 | 4,124 (random sample of the control pool, fixed seed) | 8,248 |
| non-AD dementia | 3,597 (all-cause dementia without AD; disjoint from the AD cases) | 4,137 | 7,734 |
| MCI | 756 | 4,137 | 4,893 |

The three cohorts share one pool of 4,137 controls. The 23 genes are the 20 ADSP Gene Verification Committee AD genes plus
ACE, HLA-DRB1 and PTK2B.

## `upstream_analysis/` (reference only)

These scripts need individual-level UK Biobank genotypes, phenotypes and the extracted Enformer features, and contain
cluster-specific paths. They are included to document how `data/` was produced; they are not needed to draw the figures.

How the per-gene scores behind `data/` were produced (as implemented in the scripts):

* **epiBrainLLM**: the feature/learner configuration was chosen separately for each gene by `41_maxtune.py`
  (feature set: 77 brain tracks or all 5,313 tracks; 4–16 principal components per bin; haplotypes concatenated or
  averaged; SelectKBest; logistic regression or gradient boosting), using the AD cohort (AD and non-AD dementia scores)
  or the MCI cohort (MCI scores). Seven genes without a tuned entry use `ad77_pca4 / avg / logistic C=0.3 / k=20`.
* **raw SNP**: one fixed configuration for every gene, `GradientBoostingClassifier(n_estimators=100, max_depth=3)` on all
  genotyped variants of the same 196,608-bp window.
* Both: 10-fold out-of-fold scores, combined across genes by an L2-regularised logistic stacker (C = 0.5).
* The script that wrote the AD per-gene scores (`oof_AD.npz`) was not kept as a file; it followed the same procedure as
  `make_oof_dementia.py` with the AD cohort.

| Script | Produces |
|---|---|
| `41_maxtune.py` | per-gene choice of the epiBrainLLM feature/learner configuration (`all_genes_maxtune_ALL.tsv`) |
| `make_oof_dementia.py`, `make_oof_mci_allctrl.py` | per-gene out-of-fold scores for epiBrainLLM and raw SNP (`oof_*.npz`) |
| `benchmark3_full.py` | multi-locus combine, PRS comparison, paired bootstrap → `benchmark3_source.csv`, `paired_p3.csv`, per-subject scores for the ROC curves |
| `make_gene_rank.py` | `gene_rank_*.tsv` |
| `make_curves23.py` | `seed_curve_*_full23.tsv` |
| `score_prscs.py`, `run_ldpred2.R`, `run_ldpred2.sh`, `finalize_prs.py` | polygenic scores (PRS-CS, LDpred2; SDPR was run with its command-line tool) |
