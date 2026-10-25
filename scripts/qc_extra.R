#!/usr/bin/env Rscript
# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
# 数据质控的 R 侧：WGCNA goodSamplesGenes + vegan PERMANOVA + 逐特征风格检验。
# 输入由 scripts/qc.py 写入 results/qc/。

suppressPackageStartupMessages({
  library(WGCNA)
  library(vegan)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
root <- if (length(file_arg)) {
  normalizePath(file.path(dirname(sub("^--file=", "", file_arg)), ".."))
} else {
  getwd()
}
qc_dir <- file.path(root, "results", "qc")

chem <- read.csv(file.path(qc_dir, "chem_for_r.csv"), check.names = FALSE)
sens <- read.csv(file.path(qc_dir, "sensory_for_r.csv"), check.names = FALSE)
meta <- read.csv(file.path(qc_dir, "meta_for_r.csv"), check.names = FALSE)

chem_mat <- as.matrix(chem[, setdiff(colnames(chem), "beer")])
sens_mat <- as.matrix(sens[, setdiff(colnames(sens), "beer")])

# WGCNA：标记缺失过多/零方差的样本与特征
gsg_chem <- goodSamplesGenes(chem_mat, verbose = 0)
gsg_sens <- goodSamplesGenes(sens_mat, verbose = 0)

write.csv(
  data.frame(beer = chem$beer, good_sample = gsg_chem$goodSamples,
             good_feature = ifelse(length(gsg_chem$goodGenes) == ncol(chem_mat),
                                   gsg_chem$goodGenes, NA)),
  file.path(qc_dir, "wgcna_sample_flags.csv"), row.names = FALSE
)
write.csv(
  data.frame(feature = colnames(chem_mat), good_feature = gsg_chem$goodGenes),
  file.path(qc_dir, "wgcna_feature_flags.csv"), row.names = FALSE
)
write.csv(
  data.frame(target = colnames(sens_mat), good_target = gsg_sens$goodGenes),
  file.path(qc_dir, "wgcna_target_flags.csv"), row.names = FALSE
)

# 中位数填补后标准化，供 PERMANOVA 使用
imp <- function(m) {
  for (j in seq_len(ncol(m))) {
    v <- m[, j]
    if (anyNA(v)) v[is.na(v)] <- median(v, na.rm = TRUE)
    m[, j] <- v
  }
  m
}
chem_z <- scale(imp(chem_mat))
sens_z <- scale(imp(sens_mat))

d_chem <- vegdist(chem_z, method = "euclidean")
d_sens <- vegdist(sens_z, method = "euclidean")

set.seed(42)
ad_chem <- adonis2(d_chem ~ style, data = meta, permutations = 999)
ad_sens <- adonis2(d_sens ~ style, data = meta, permutations = 999)

bd_chem <- betadisper(d_chem, meta$style)
bd_sens <- betadisper(d_sens, meta$style)
pt_chem <- permutest(bd_chem, permutations = 999)
pt_sens <- permutest(bd_sens, permutations = 999)

# 逐特征风格差异：Kruskal-Wallis + BH
kw <- function(m) {
  p <- apply(m, 2, function(v) {
    if (length(unique(v[!is.na(v)])) < 2) return(NA_real_)
    tryCatch(kruskal.test(v ~ meta$style)$p.value, error = function(e) NA_real_)
  })
  data.frame(feature = colnames(m), p = p, q = p.adjust(p, "BH"))
}
kw_chem <- kw(chem_mat)
kw_sens <- kw(sens_mat)
write.csv(kw_chem, file.path(qc_dir, "style_feature_tests_chem.csv"), row.names = FALSE)
write.csv(kw_sens, file.path(qc_dir, "style_feature_tests_sens.csv"), row.names = FALSE)

top_chem <- head(kw_chem[order(kw_chem$q), ], 15)
top_sens <- head(kw_sens[order(kw_sens$q), ], 15)

out <- list(
  wgcna = list(
    chem_good_samples = sum(gsg_chem$goodSamples),
    chem_good_features = sum(gsg_chem$goodGenes),
    sens_good_samples = sum(gsg_sens$goodSamples),
    sens_good_targets = sum(gsg_sens$goodGenes)
  ),
  permanova_chem = list(
    r2 = ad_chem$R2[1], f = ad_chem$F[1], p = ad_chem$`Pr(>F)`[1]
  ),
  permanova_sens = list(
    r2 = ad_sens$R2[1], f = ad_sens$F[1], p = ad_sens$`Pr(>F)`[1]
  ),
  betadisper_chem_p = pt_chem$tab$`Pr(>F)`[1],
  betadisper_sens_p = pt_sens$tab$`Pr(>F)`[1],
  style_kruskal = list(
    chem_n_fdr05 = sum(kw_chem$q < 0.05, na.rm = TRUE),
    sens_n_fdr05 = sum(kw_sens$q < 0.05, na.rm = TRUE),
    top_chem = top_chem,
    top_sens = top_sens
  )
)
write_json(out, file.path(qc_dir, "r_extra.json"), auto_unbox = TRUE, pretty = TRUE, digits = 6)
cat("[QC-R] done: results/qc/r_extra.json\n")
