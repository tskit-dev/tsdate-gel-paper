df <- read.csv(zstdfile("data/sampling_sim_data+60000.csv.zst"), sep="\t")
df$bal  <- df$sampling_scheme...unbalanced....YRI..2400..CEU..57000..CHB..600...balanced....YRI..20000..CEU..20000..CHB..20000..
df$pop <- factor(df$pop...YRI..0..CEU..1..CHB..2.)
df.u.rare <- subset(df, freq <= 0.001)

par(mfrow = c(2, 4))
line = list(b = -1.5, u = -23)
data = list(all.variants = df, ultra.rare = df.u.rare)

for (bal in c('b', 'u')) {
    b_str = ifelse(bal == 'b', 'Balanced', 'Unbalanced')
    cat(b_str, "sampling\n")
    for (include_pop in c(FALSE, TRUE)) {
        show_title <- TRUE
        for (age in c('true_midtime', 'inferred_time')) {
            for (name in names(data)) {
                tmp_df <- data[[name]]
                if (include_pop) {
                    m_full <- lm(s ~ log(get(age)) + log(freq) + pop, data=tmp_df[tmp_df$bal == bal,])
                } else {
                    m_full <- lm(s ~ log(get(age)) + log(freq), data=tmp_df[tmp_df$bal == bal,])
                }
                str_model <- deparse(formula(m_full))
                title <- paste0(name, " ", age, " ", str_model)
                # Uncomment to plot residuals and other model diagnostics
                #plot(m_full, ask = FALSE, sub.caption = "")
                #mtext(title, side = 3, line = line[[bal]], outer = TRUE, cex = 1.0, font = 2)
                m_no_age  <- update(m_full, . ~ . - log(get(age)))
                m_no_freq <- update(m_full, . ~ . - log(freq))
                if (show_title) {
                    cat("Model:", str_model, "\n")
                    show_title <- FALSE
                }
                logtime  <- paste0("log(", age, ")")
                pR2_age  <- 1 - deviance(m_full) / deviance(m_no_age)
                pR2_freq <- 1 - deviance(m_full) / deviance(m_no_freq)
                cat(
                    name, "partialR2 for", logtime, "vs log(freq): ",
                    pR2_age, "vs", pR2_freq, "(ratio:", pR2_age / pR2_freq, ")\n"
                )
            }
            cat("\n")
        }
    }
}

# LaTeX table output
{
    cat("\\begin{table}[h]\n")
    cat("\\centering\n")
    cat("\\caption{Proportional reduction in deviance: age vs. frequency}\n")
    cat("\\label{tab:sim_deviance_analysis}\n")
    cat("\\begin{tabular}{lllrrr}\n")
    cat("\\hline\n")
    cat(
        "\\textbf{Sampling} & \\textbf{Model} & \\textbf{Data} &",
        "\\textbf{Age $R^2$} & \\textbf{Freq $R^2$} & \\textbf{Ratio} \\\\\n")

    for (bal in c('b', 'u')) {
        prefix <- "\\hline\n"
        b_str = ifelse(bal == 'b', 'Balanced', 'Unbalanced')
        line <- (paste0("{\\textit{", b_str, "}} "))
        for (include_pop in c(FALSE, TRUE)) {
            for (age in c('true_midtime', 'inferred_time')) {
                age_str = ifelse(age == 'true_midtime', '$t_\\text{true}$', '$t_\\text{tsdate}$')
                line <- paste0(
                    line,
                    "& $s \\sim \\log(\\text{", age_str, "}) + \\log(\\text{freq})",
                    ifelse(include_pop, " + \\text{pop}", ""),
                    "$ ")
                cat(prefix)
                cat(line)
                prefix <- "\\cline{2-6}\n"
                for (name in names(data)) {
                    tmp_df <- data[[name]]
                    if (include_pop) {
                        m_full <- lm(s ~ log(get(age)) + log(freq) + pop, data=tmp_df[tmp_df$bal == bal,])
                    } else {
                        m_full <- lm(s ~ log(get(age)) + log(freq), data=tmp_df[tmp_df$bal == bal,])
                    }

                    m_no_age  <- update(m_full, . ~ . - log(get(age)))
                    m_no_freq <- update(m_full, . ~ . - log(freq))
                    logtime  <- paste0("log(", age, ")")
                    pR2_age  <- 1 - deviance(m_full) / deviance(m_no_age)
                    pR2_freq <- 1 - deviance(m_full) / deviance(m_no_freq)
                    cat(
                        ifelse(line == "", "& &", "&"),
                        ifelse(name == "all.variants", "all", "ultra-rare"),
                        "&", pR2_age, "&", pR2_freq, "&", pR2_age / pR2_freq, "\\\\\n"
                    )
                    line <- ""
                }
            }
        }
    }

    cat("\\hline\n")
    cat("\\end{tabular}\n")
    cat("\\end{table}\n")
}