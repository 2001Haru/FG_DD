args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: Rscript mixed_model_post_eval.R <paired_gains.csv> <output.json> <output.csv>")
}

required <- c("lme4", "lmerTest", "jsonlite")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop(paste0(
    "missing R packages: ", paste(missing, collapse = ", "),
    ". Install lme4, lmerTest, and jsonlite; no z-test fallback is used."
  ))
}

input_path <- args[[1]]
json_path <- args[[2]]
csv_path <- args[[3]]
data <- read.csv(input_path, stringsAsFactors = FALSE)
required_columns <- c("comparison", "recovery_seed", "student_seed", "paired_gain")
if (!all(required_columns %in% names(data))) {
  stop("input CSV does not contain comparison/recovery_seed/student_seed/paired_gain")
}

fit_one <- function(frame) {
  frame$recovery_seed <- factor(frame$recovery_seed)
  frame$student_seed <- factor(frame$student_seed)
  if (nlevels(frame$recovery_seed) < 2 || nlevels(frame$student_seed) < 2) {
    stop("at least two recovery and two student seed levels are required")
  }
  contrasts(frame$student_seed) <- contr.sum(nlevels(frame$student_seed))
  model <- lmerTest::lmer(
    paired_gain ~ student_seed + (1 | recovery_seed),
    data = frame,
    REML = TRUE,
    control = lme4::lmerControl(
      optimizer = "bobyqa",
      check.conv.singular = "ignore"
    )
  )
  coefficient_table <- summary(model, ddf = "Satterthwaite")$coefficients
  intercept <- coefficient_table["(Intercept)", ]
  variance_table <- as.data.frame(lme4::VarCorr(model))
  recovery_variance <- variance_table$vcov[variance_table$grp == "recovery_seed"][[1]]
  residual_variance <- variance_table$vcov[variance_table$grp == "Residual"][[1]]
  total_variance <- recovery_variance + residual_variance

  recovery_means <- aggregate(
    paired_gain ~ recovery_seed, data = frame, FUN = mean
  )$paired_gain
  naive <- t.test(recovery_means, mu = 0)
  fixed_effects <- lme4::fixef(model)
  student_effects <- fixed_effects[names(fixed_effects) != "(Intercept)"]
  optimizer_messages <- model@optinfo$conv$lme4$messages
  if (is.null(optimizer_messages)) optimizer_messages <- character(0)

  list(
    estimate = unname(intercept[["Estimate"]]),
    standard_error = unname(intercept[["Std. Error"]]),
    t = unname(intercept[["t value"]]),
    denominator_df_satterthwaite = unname(intercept[["df"]]),
    two_sided_p_satterthwaite = unname(intercept[["Pr(>|t|)"]]),
    recovery_random_intercept_variance = unname(recovery_variance),
    residual_variance = unname(residual_variance),
    intraclass_correlation = if (total_variance > 0) {
      unname(recovery_variance / total_variance)
    } else {
      NA_real_
    },
    singular_fit = lme4::isSingular(model, tol = 1e-4),
    optimizer_messages = as.list(optimizer_messages),
    student_seed_fixed_effects_sum_contrast = as.list(student_effects),
    recovery_seed_levels = as.list(levels(frame$recovery_seed)),
    student_seed_levels = as.list(levels(frame$student_seed)),
    naive_recovery_mean_t_check = list(
      recovery_seed_means = as.list(unname(recovery_means)),
      estimate = unname(mean(recovery_means)),
      sample_sd = unname(sd(recovery_means)),
      t = unname(naive$statistic),
      df = unname(naive$parameter),
      two_sided_p = unname(naive$p.value)
    )
  )
}

comparisons <- sort(unique(data$comparison))
results <- list(
  model = paste0(
    "gain ~ 1 + student_seed (sum-to-zero fixed block) + ",
    "(1 | recovery_seed), REML"
  ),
  df_method = "lmerTest Satterthwaite",
  note = paste0(
    "The intercept is the balanced grand mean because student_seed uses sum contrasts. ",
    "Cells sharing a recovery seed are modeled by a random intercept."
  ),
  comparisons = list()
)

summary_rows <- list()
for (comparison_name in comparisons) {
  frame <- data[data$comparison == comparison_name, ]
  result <- fit_one(frame)
  results$comparisons[[comparison_name]] <- result
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    comparison = comparison_name,
    estimate = result$estimate,
    standard_error = result$standard_error,
    t = result$t,
    denominator_df_satterthwaite = result$denominator_df_satterthwaite,
    two_sided_p_satterthwaite = result$two_sided_p_satterthwaite,
    recovery_random_intercept_variance = result$recovery_random_intercept_variance,
    residual_variance = result$residual_variance,
    intraclass_correlation = result$intraclass_correlation,
    singular_fit = result$singular_fit,
    naive_t = result$naive_recovery_mean_t_check$t,
    naive_df = result$naive_recovery_mean_t_check$df,
    naive_two_sided_p = result$naive_recovery_mean_t_check$two_sided_p
  )
}

dir.create(dirname(json_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(results, json_path, pretty = TRUE, auto_unbox = TRUE, digits = NA)
write.csv(do.call(rbind, summary_rows), csv_path, row.names = FALSE)
cat(jsonlite::toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = NA), "\n")
