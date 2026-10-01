# site-specific paths (set as environment variables before running)
EPIBRAIN_ROOT=${EPIBRAIN_ROOT:-$(pwd)}
suppressMessages(library(bigsnpr))
options(bigstatsr.check.parallel.blas = FALSE, default.nproc.blas = NULL)
NCORES <- 4
base <- "${EPIBRAIN_ROOT}/UKB_section/01_benchmark_PRS"
setwd(base); dir.create("ldpred2_tmp", showWarnings=FALSE)

ss <- bigreadr::fread2("sumstats/kunkle_ldpred2.txt")  # rsid chr pos a1 a0 beta beta_se n_eff p
cat("sumstats rows:", nrow(ss), "\n")

corr <- NULL; df_beta_all <- NULL; ld <- numeric(0)
tmp_corr <- file.path("ldpred2_tmp", "corr")
if (file.exists(paste0(tmp_corr,".sbk"))) file.remove(paste0(tmp_corr,".sbk"))

for (chr in 1:22) {
  bedf <- paste0("ref/ukb_chr", chr, ".bed"); rdsf <- sub("\\.bed$", ".rds", bedf)
  if (!file.exists(rdsf)) snp_readBed2(bedf, ncores=1)
  obj <- snp_attach(rdsf); Gc <- obj$genotypes
  mapc <- data.frame(chr=obj$map$chromosome, pos=obj$map$physical.pos,
                     a0=obj$map$allele2, a1=obj$map$allele1, rsid=obj$map$marker.ID, stringsAsFactors=FALSE)
  ssc <- ss[ss$chr == chr, ]
  sumc <- data.frame(chr=ssc$chr, pos=ssc$pos, a0=ssc$a0, a1=ssc$a1,
                     beta=ssc$beta, beta_se=ssc$beta_se, n_eff=ssc$n_eff, rsid=ssc$rsid, stringsAsFactors=FALSE)
  matched <- tryCatch(snp_match(sumc, mapc, join_by_pos=FALSE), error=function(e){cat("match err chr",chr,conditionMessage(e),"\n");NULL})
  if (is.null(matched) || nrow(matched)==0) next
  POS2 <- tryCatch(snp_asGeneticPos(matched$chr, matched$pos, dir="ldpred2_tmp", ncores=1),
                   error=function(e){cat("genpos fallback chr",chr,"\n"); matched$pos/1e6})
  ind <- matched[["_NUM_ID_"]]
  corr0 <- snp_cor(Gc, ind.col=ind, size=3/1000, infos.pos=POS2, ncores=NCORES)
  ld <- c(ld, Matrix::colSums(corr0^2))
  if (is.null(corr)) corr <- as_SFBM(corr0, tmp_corr, compact=TRUE) else corr$add_columns(corr0, nrow(corr))
  df_beta_all <- rbind(df_beta_all, matched)
  cat("chr", chr, "matched", nrow(matched), "| corr ncol", ncol(corr), "\n"); flush.console()
}
saveRDS(df_beta_all, "ldpred2_tmp/df_beta.rds")
cat("TOTAL matched SNPs:", nrow(df_beta_all), "\n")

ldsc <- snp_ldsc(ld, ld_size=length(ld), chi2=(df_beta_all$beta/df_beta_all$beta_se)^2,
                 sample_size=df_beta_all$n_eff, ncores=NCORES)
h2_est <- max(ldsc[["h2"]], 0.001); cat("LDSC h2:", ldsc[["h2"]], "-> used", h2_est, "\n")

set.seed(1)
multi_auto <- snp_ldpred2_auto(corr, df_beta_all, h2_init=h2_est,
                               vec_p_init=seq_log(1e-4, 0.2, length.out=30),
                               ncores=NCORES, allow_jump_sign=FALSE, shrink_corr=0.95)
rg <- sapply(multi_auto, function(a) diff(range(a$corr_est)))
keep <- rg > (0.95 * quantile(rg, 0.95, na.rm=TRUE)); keep[is.na(keep)] <- FALSE
if (sum(keep)==0) keep <- rep(TRUE, length(multi_auto))
beta_auto <- rowMeans(sapply(multi_auto[keep], function(a) a$beta_est))

out <- data.frame(SNP=df_beta_all$rsid, A1=df_beta_all$a1, BETA=beta_auto)
out <- out[is.finite(out$BETA) & out$BETA != 0, ]
write.table(out, "result/ldpred2_betas_all.txt", row.names=FALSE, col.names=FALSE, quote=FALSE)
cat("WROTE", nrow(out), "LDpred2 betas\nLDPRED2_DONE\n")
