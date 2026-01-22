
hex_scatter <- function(
  data,
  x, y,                         # tidy-eval cols, e.g. x = mean_time, y = AF_ts
  xlab = NULL, ylab = NULL,
  log_x = TRUE, log_y = TRUE,
  bins = 100,
  show_cor = TRUE,
  cor_method = c("pearson","spearman","kendall"),
  cor_label_x = NULL, cor_label_y = NULL,
  cor_coords = c("log10","native"),  # <- pass coords like you used to
  cor_size = 5,
  palette = "plasma"
) {
  cor_method <- match.arg(cor_method)
  cor_coords <- match.arg(cor_coords)

  x_sym <- rlang::ensym(x); y_sym <- rlang::ensym(y)

  # Collect minimal data (works for data.frame/tibble or r-polars LazyFrame)
  if (inherits(data, "RPolarsLazyFrame") || inherits(data, "LazyFrame")) {
    df <- data$select(rlang::as_string(x_sym), rlang::as_string(y_sym))$collect()
  } else {
    df <- dplyr::select(as.data.frame(data), !!x_sym, !!y_sym)
  }
  names(df) <- c("x","y")

  # Keep finite; require positives on log axes
  ok <- is.finite(df$x) & is.finite(df$y)
  if (log_x) ok <- ok & df$x > 0
  if (log_y) ok <- ok & df$y > 0
  df <- df[ok, , drop = FALSE]
  if (!nrow(df)) stop("No finite (and positive, if log) values in x/y.")

  # Labels
  if (is.null(xlab)) xlab <- rlang::as_name(x_sym)
  if (is.null(ylab)) ylab <- rlang::as_name(y_sym)

  # Correlation computed in transformed space (if log axes)
  x_for_cor <- if (log_x) log10(df$x) else df$x
  y_for_cor <- if (log_y) log10(df$y) else df$y
  ct <- stats::cor.test(x_for_cor, y_for_cor, method = cor_method)
  r  <- unname(ct$estimate); p <- ct$p.value
  cor_lab <- paste0("r = ", formatC(r, format = "f", digits = 3),
                    "\np = ", format.pval(p, digits = 2, eps = 1e-300))

  # Default text position if not supplied
  if (is.null(cor_label_x) || is.null(cor_label_y)) {
    qx <- stats::quantile(df$x, c(0.08), na.rm = TRUE)
    qy <- stats::quantile(df$y, c(0.08), na.rm = TRUE)
    if (cor_coords == "log10") {
      cor_label_x <- if (log_x) log10(qx[[1]]) else qx[[1]]
      cor_label_y <- if (log_y) log10(qy[[1]]) else qy[[1]]
    } else {
      cor_label_x <- qx[[1]]
      cor_label_y <- qy[[1]]
    }
  }

  # Build plot
  library(ggplot2)
  p <- ggplot(df, aes(x = x, y = y)) +
    geom_hex(bins = bins) +
    scale_fill_viridis_c(option = palette, trans = "log") +
    labs(x = xlab, y = ylab) +
    theme_classic(base_size = 20) +
    theme(legend.position = "none",
          plot.margin = grid::unit(c(0,0,0,0), "lines"),
          strip.background = element_blank())

  # Axes
  if (log_x) {
    p <- p + scale_x_log10(
      breaks = scales::trans_breaks("log10", function(z) 10^z),
      labels = scales::trans_format("log10", scales::math_format(10^.x))
    ) + annotation_logticks(sides = "b", linewidth = 0.5, colour = "grey20")
  }
  if (log_y) {
    p <- p + scale_y_log10(
      breaks = scales::trans_breaks("log10", function(z) 10^z),
      labels = scales::trans_format("log10", scales::math_format(10^.x))
    ) + annotation_logticks(sides = "l", linewidth = 0.5, colour = "grey20")
  }

  # Correlation label at user coords (either native or log10 space)
  if (show_cor) {
    # convert coords to native axis space if user supplied log10 coords
    x_lab_native <- if (cor_coords == "log10" && log_x) 10^cor_label_x else cor_label_x
    y_lab_native <- if (cor_coords == "log10" && log_y) 10^cor_label_y else cor_label_y
    p <- p + annotate("text", x = x_lab_native, y = y_lab_native,
                      label = cor_lab, hjust = 0, vjust = 0,
                      size = cor_size, colour = "black")
  }

  p
}

## Figure 5
pop_strat_bar_plot <- function(
  group_numbers,
  count_col = "n",
  scale_by = 1000,
  base_size_num = 20,
  legend_title = "PCA group",
  legend_text_size = 14,
  legend_title_size = 16
) {
  count_sym <- rlang::sym(count_col)

  ggplot(
    group_numbers,
    aes(x = Dataset, y = {{ count_sym }}/scale_by, fill = pop)
  ) +
    geom_bar(stat = "identity", colour = NA) +  # drop colour mapping to avoid 2 legends
    theme_classic(base_size = base_size_num) +
    scale_y_continuous(expand = c(0, 0)) +
    theme(
      axis.line.x  = element_blank(),
      axis.ticks.x = element_blank(),
      legend.text  = element_text(size = legend_text_size),
      legend.title = element_text(size = legend_title_size)
    ) +
    guides(
      fill = guide_legend(
        title = legend_title,
        nrow = 1,
        reverse = TRUE
      )
    )
}

geomean_allele_ages <- function(
  df,
  group_col = "AC_ts",
  group_vals = c(1:10),
  time_col = "mean_time",
  strat_col = NULL,
  filt_cols = NULL,
  filt_vals = NULL
) {
  
  group_sym <- rlang::sym(group_col)
  time_sym <- rlang::sym(time_col)
  df_filt <- df |>
    filter({{group_sym}} %in% group_vals) |>
    multi_filt_cols(
      filt_cols = filt_cols,
      filt_vals = filt_vals
    )
  if (!is.null(strat_col)) {
    strat_sym <- rlang::sym(strat_col)
    geomean_ages <- df_filt |>
      drop_na({{strat_sym}}) |>
      group_by({{group_sym}}, {{strat_sym}}) |>
      summarise(geomean_allele_age = exp(mean(log({{time_sym}})))) |>
      ungroup()
  } else {
    geomean_ages <- df_filt |>
      group_by({{group_sym}}) |>
      summarise(geomean_allele_age = exp(mean(log({{time_sym}})))) |>
      ungroup()
  }
  return(geomean_ages)
}

ecdf_grouped_plot <- function(
  df,
  time_col = "mean_time",
  group_col = "AC_ts",
  group_vals = NULL,
  filt_cols = NULL,
  filt_vals = NULL,
  discrete = FALSE
) {
  
  group_sym <- rlang::sym(group_col)
  time_sym <- rlang::sym(time_col)  

  df_filt <- df |>
    drop_na({{time_sym}}, {{group_sym}}) |>
    multi_filt_cols(
      filt_cols = filt_cols,
      filt_vals = filt_vals
    ) |>
    filter(
      {{group_sym}} %in% group_vals
    )
  
  geomean_df <- df_filt |>
    geomean_allele_ages(
      time_col = time_col,
      group_col = group_col,
      group_vals = group_vals
    )

  if (discrete) {
    df_filt %<>%
      mutate({{group_sym}} := factor({{group_sym}}, levels = as.character(group_vals)))
    geomean_df %<>%
      mutate({{group_sym}} := factor({{group_sym}}, levels = as.character(group_vals)))
    print(geomean_df)
  }

  df_filt |>
    ggplot(
      aes(
      x = log10({{time_sym}}),
      group = {{group_sym}},
      colour = {{group_sym}}
    )
  ) +
  theme_classic(base_size = 20) +
  theme(
    strip.background = element_blank()
  ) +
  stat_ecdf(
    geom = "smooth",
    linewidth = 0.5
  ) +
  geom_vline(
    data = geomean_df,
    linetype = "dotted",
    linewidth = 0.4,
    aes(
      xintercept = log10(geomean_allele_age),
      colour = {{group_sym}}
    )
  ) +
  scale_y_continuous(
    expand = c(0, 0),
    breaks = c(0, 0.5, 1)
  ) +
  annotation_logticks(
    linewidth = 0.5,
    colour = "grey20",
    short = unit(0.05, "cm"),
    mid = unit(0.075, "cm"),
    long = unit(0.1, "cm"),
    sides = "b"
  ) +
  scale_x_continuous(
    limits = c(0, 4),
    labels = scales::math_format(10^.x)
  )
}

plot_coalescence <- function(data, pop_means, order_vec, facet = FALSE, output_file, max_gen = 100000, pop_colours_vec) {
  pop_means %<>%
    filter(start_time <= max_gen) %>%
    mutate(pop = factor(pop, levels = order_vec))
  p <- data %>%
    mutate(pop = factor(pop, levels = order_vec)) %>%
    filter(start_time <= max_gen) %>%
    ggplot(aes(x = log10(start_time), y = log10(rolling_avg), group = individual_id, colour = pop)) +
    geom_line(alpha = 0.04, linewidth = 0.05) +
    theme_classic(base_size = 20) +
    geom_line(data = pop_means, aes(x = log10(start_time), y = log10(pop_avg), group = pop, colour = pop), 
              linewidth = 0.5, linetype = "dashed") +
    scale_color_manual(values = pop_colours_vec) +
    annotation_logticks(
      linewidth = 0.5,
      colour = "grey20",
      short = unit(0.05, "cm"),
      mid = unit(0.075, "cm"),
      long = unit(0.1, "cm"),
      sides = "bl"
    ) +
    scale_x_continuous(
      labels = scales::math_format(10^.x)
    ) +
    scale_y_continuous(
      labels = scales::math_format(10^.x)
    ) +
    labs(x = "Time (generations ago)", y = "Coalescence counts", colour = "PCA group") +
    theme(legend.position = 'NONE')

  if (facet) {
    p <- p + facet_wrap(~pop)
  }
  return(p)
}

grouped_density_plot <- function(
  df,
  time_col = "mean_time",
  group_col = "AC_ts",
  group_vals = c(1:9),
  filt_cols = NULL,
  filt_vals = NULL,
  log10 = TRUE,
  flip = FALSE,
  discrete = FALSE
) {

  time_sym <- rlang::sym(time_col)
  group_sym <- rlang::sym(group_col)
  
  ## Filter columns
  df_filt <- df |>
  drop_na({{time_sym}}, {{group_sym}}) |>
  multi_filt_cols(
    filt_cols = filt_cols,
    filt_vals = filt_vals
  ) |>
  filter({{group_sym}} %in% group_vals)

  if (log10) {
    df_filt %<>% mutate({{time_sym}} := log10({{time_sym}}))
  }

  if (discrete) {
    df_filt %<>% mutate({{group_sym}} := factor({{group_sym}}, levels = group_vals))
  }

  if(flip) {
    plot <- df_filt |>
      ggplot(
        aes(
          y = {{time_sym}},
          group = {{group_sym}},
          colour = {{group_sym}}
        )
      )
  } else {
    plot <- df_filt |>
      ggplot(
        aes(
          x = {{time_sym}},
          group = {{group_sym}},
          colour = {{group_sym}}
        )
      )
  }

  plot <- plot +
  geom_density() +
  theme_classic() +
  scale_y_continuous(expand = c(0,0))

  if (log10) {
    plot <- plot +
      annotation_logticks(
        linewidth = 0.5,
        colour = "grey20",
        short = unit(0.05, "cm"),
        mid = unit(0.075, "cm"),
        long = unit(0.1, "cm"),
        sides = "b"
      ) +
      scale_x_continuous(
        labels = scales::math_format(10^.x),
        limits = c(log10(0.001), log10(1000))
      ) 
  }
  return(plot)
}

create_joint_density_plot <- function(
  to_plot,
  highlight_id = NULL,
  sample_name,
  group_numbers,
  pop_colours_vec,
  bins = 100,
  smooth_method = "lm",
  smooth_linetype = "dotted",
  smooth_colour = "black",
  smooth_linewidth = 0.4,
  cor_method = "pearson",
  cor_size = 3,
  cor_label_x,
  cor_label_y,
  include_group_hists = FALSE,
  cor_colour = "black"
) {

  # Create central plot
  combined_plot <- to_plot |>
    ggplot(aes(
      x = geometric_mean_age_singletons,
      y = number_of_singletons,
    )) +
    geom_hex(bins = bins) +
    theme_classic(base_size = 20) +
    theme(
      legend.position = "NONE",
      #plot.margin = unit(c(0, 0, 0, 0), 'lines')
    ) +
    scale_fill_viridis(option = "plasma", trans = "log") +
    labs(
      x = "Geometric mean singleton age (generations)",
      y = expression("Number of singletons")
    ) +
    geom_smooth(
      method = smooth_method,
      se = FALSE,
      linewidth = smooth_linewidth,
      colour = smooth_colour,
      linetype = smooth_linetype
    ) +
    stat_cor(
      method = "spearman",
      #cor.coef.name = "Spearmans rank",
      label.x = cor_label_x,
      label.y = cor_label_y,
      size = cor_size,
      colour = cor_colour
    ) +
    scale_x_continuous(expand = c(0,0), limits = c(0, 220)) +
    scale_y_continuous(limits = c(0, NA), expand = c(0, 0))

  if (!is.null(highlight_id)) {
    to_plot |> filter(singleton_sample_id == highlight_id) -> to_highlight
    combined_plot <- combined_plot +
    geom_point(
      data = to_highlight,
      shape = 21,
      size = 7,
      stroke = 1.1,
      colour = "black"
    ) +
    geom_text(
      data = to_highlight,
      label = sample_name,
      colour = "black",
      size = 7,
      hjust = -0.2, vjust = -0.1
    )
  }

  if(include_group_hists) {
    # Create xplot
  xplot <- to_plot |>
    grouped_density_plot(
      time_col = "geometric_mean_age_singletons",
      group_col = "pop",
      group_vals = group_numbers$pop,
      log10 = FALSE,
      flip = FALSE
    ) +
    scale_colour_manual(values = pop_colours_vec) +
    theme_void() +
    theme(
      legend.position = "NONE",
      plot.margin = unit(c(0, 0, 0, 0), 'lines')
    )

  # Create yplot
  yplot <- to_plot |>
    grouped_density_plot(
      time_col = "number_of_singletons",
      group_col = "pop",
      group_vals = group_numbers$pop,
      log10 = FALSE,
      flip = TRUE
    ) +
    scale_colour_manual(values = pop_colours_vec) +
    theme_void() +
    theme(
      legend.position = "NONE",
      plot.margin = unit(c(0, 0, 0, 0), 'lines')
    )

  # Combine plots
  combined_plot <- ggarrange(
    xplot, NULL,
    combined_plot, yplot,
    ncol = 2, nrow = 2,
    align = "hv",
    widths = c(1, 0.6),
    heights = c(0.4, 1)
  )
  }

  return(combined_plot)
}

adac_along_haps_plot <- function(
  adac_haps,
  adac_col = "ADAC_ts",
  haplotype_col = "mutation_hap",
  AC_thresh = 1,
  AC_highlight = 1,
  haplotype_number = "Haplotype 1",
  remove_x_axis = TRUE,
  rolling_average_nmutations = 10
) {
  adac_sym <- rlang::sym(adac_col)
  haplotype_sym <- rlang::sym(haplotype_col)
  
  # Prepare data for plotting
  rolling_avg <- adac_haps |>
    filter(AC_ts < AC_thresh & mutation_hap == haplotype_number) |>
    distinct() |>
    mutate(
      newcol = zoo::rollapply(
        {{adac_sym}},
        rolling_average_nmutations,
        mean,
        fill = NA
      )
    ) |>
    arrange(newcol) |>
    ungroup()
  
  AF_label = paste0("DAF <" , round(((AC_thresh / 95070) * 100), 2), "%")
  # Create the plot
  plot <- adac_haps |>
    filter(AC_ts %in% AC_highlight & mutation_hap == haplotype_number) |>
    ggplot(aes(x = position / 1e6, y = {{adac_sym}})) +
    
    # Highlighted points with color in the legend
    geom_point(aes(color = "GEL DAF < 0.1%"), size = 0.125) +
    
    # Rolling average line with color in the legend
    geom_line(
      data = rolling_avg,
      aes(y = newcol, color = "Rolling average (10 alleles)"),
      size = 0.4
    ) +

    # Theme and labels
    theme_bw(base_size = 20) +
    labs(
      x = "",
      y = "z-score*",
      color = NULL  # Removes the legend title
    ) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position = "top",          # Moves legend to the top of the plot
      legend.justification = "center"
    ) +
    
    # Scales
    scale_x_continuous(expand = c(0, 0)) +
    scale_color_manual(
      values = c("Rolling average (10 alleles)" = "darkred", "GEL DAF < 0.1%" = "grey")
    ) +
    coord_cartesian(clip = "off") +
    
    # Upward arrow inside the plot
    annotate(
      "segment",
      x = 1.5, xend = 1.5,  # Arrow positioned at x = 1.5
      y = 0.05, yend = 5,   # Upward arrow from y = 0.05 to y = 3
      arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
      colour = "grey20"
    ) +

    # Downward arrow inside the plot
    annotate(
      "segment",
      x = 1.5, xend = 1.5,  # Arrow positioned at x = 1.5
      y = -0.05, yend = -5, # Downward arrow from y = -0.05 to y = -3
      arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
      colour = "grey20"
    )

  # Optionally remove x-axis
  if (remove_x_axis) {
    plot <- plot +
      theme(
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank()
      )
  }
  
  return(plot)
}

gnn_along_haps_plot <- function(
  gnn_haps,
  pop_vec,
  haplotype_col = "haplotype",
  haplotype_number = 1,
  remove_x_axis = TRUE
) {

  haplotype_sym <- rlang::sym(haplotype_col)
  plot <- gnn_haps |>
    filter({{haplotype_sym }} == haplotype_number) |>
    mutate(pop = factor(pop, levels = pop_vec)) |>
    ggplot() +
    geom_rect(
    aes(
      xmin = start/1000000, xmax = end/1000000,
      ymin = ymin, ymax = ymax,
      colour = pop, fill = pop
      ) 
    ) +
    theme_bw(base_size = 20) +
    scale_y_continuous(expand = c(0,0), breaks = c(0, 0.5, 1)) +
    scale_x_continuous(expand = c(0,0)) +
    theme(
      legend.position = "none",
      panel.grid = element_blank()
    ) +
    labs(
      x = "",
      y = "GNN"
    )

  if (remove_x_axis) {
    plot <- plot +
      theme(
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank()
      )
  }
  return(plot)
}

x_pos_only_plot <- function(
  p,
  xaxis_title = "Position (Mb)\nChromosome 17"
) {
  ggplot(p, aes(x = position/1000000, y = position)) +
  geom_point() +
  scale_y_continuous(limits = c(0, 0)) +
  theme_minimal(base_size = 20) +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank()
  ) +
  xlab(xaxis_title)
}

# t, r, b, l
stitch_haps_plot <- function(
  gnn_haps1_plot,
  gnn_haps2_plot,
  adac_haps1_plot,
  adac_haps2_plot,
  x_only_plot,
  sample_name = "WB01"
) {

  gnn_haps1 <- gnn_haps1_plot +
  theme(plot.margin = unit(c(0,0.15,0.15,0), 'lines'))
  gnn_haps2 <- gnn_haps2_plot +
  theme(plot.margin = unit(c(0,0.15,0.15,0), 'lines'))
  adac_haps1 <- adac_haps1_plot +
  theme(
    plot.margin = unit(c(0,0.15,0.15,0), 'lines'),
    legend.position = "NONE"
  )
  adac_haps2 <- adac_haps2_plot +
  theme(
    plot.margin = unit(c(0,0.15,0.15,0), 'lines'),
    legend.position = "NONE"
  )
  haps_xaxis <- x_only_plot +
  theme(plot.margin = unit(c(0,0.15,0.15,0), 'lines'))

  hap1 <- ggarrange(
    adac_haps1,
    gnn_haps1,
    ncol = 1,
    align = "v",
    heights = c(1, 0.33)
  )

  hap2 <- ggarrange(
    adac_haps2,
    gnn_haps2,
    NULL,
    haps_xaxis,
    ncol = 1,
    align = "v",
    heights = c(1, 0.33, 0.1, 0.225)
  )

  legend <- get_legend(adac_haps1_plot)

  # Create the combined plot
  plot <- plot_grid(
  plot_grid(
    NULL,
    ggdraw() +
      draw_label(
        sample_name, 
        x = 0.1, 
        y = 0.5, 
        hjust = 0, 
        size = 20
      ),
    NULL,
    legend,
    nrow = 1,
    rel_widths = c(0.075, 0.1, 0.2, 0.5), # Adjust as needed for spacing
    align = "hv"
  ),
  plot_grid(
    hap1, NULL, hap2,
    ncol = 1,
    rel_heights = c(1, 0.025, 1.325)
  ),
  ncol = 1,
  rel_heights = c(0.075, 1)
)

  return(plot)
}

gather_afs <- function(
  acs_df,
  positions_vec,
  exclude_pops = c("remaining"),
  datasets_vec
) {
  acs_df |>
  filter(position %in% positions_vec) |>
    pivot_longer(
      !c(position),
      names_to = "name",
      values_to = "value"
    ) |>
    mutate(
      ac_an = case_when(
        grepl("AN", name) ~ "AN",
        grepl("AC", name) ~ "AC"
      ),
      name = gsub("AN_", "", gsub("AC_", "", name))
    ) |>
    drop_na(ac_an) |>
    pivot_wider(
      names_from = "ac_an",
      values_from = "value"
    ) |>
    mutate(
      AF = AC / AN,
      AF = ifelse(is.na(AF), 0, AF)
    ) |>
    separate(
      name,
      sep = "_",
      into = c("pop", "dataset", "type")
    ) |>
    filter(
      !(pop %in% exclude_pops),
      dataset %in% datasets_vec
    ) |>
    drop_na(type) |>
    mutate(
      AF_log = ifelse(log10(AF) == -Inf, NA, log10(AF))
    )
}

af_heatmap_plot <- function(
  pop_af_df,
  AF_col = "AF_log",
  label = "DAF",
  empty = FALSE
) {
  AF_sym <- rlang::sym(AF_col)
  plot <- pop_af_df |>
    drop_na(AF_col) |>
    filter(AF_col > 0) |>
    ggplot(aes(
      x = pop,
      y = position,
      colour = {{AF_sym}},
      fill = {{AF_sym}}
    )) +
    theme_classic(base_size = 20) +
    facet_wrap(~dataset, strip.position = "bottom") +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      axis.ticks  = element_blank(),
      axis.line.y = element_blank(),
      axis.title.x = element_blank(),
      #strip.text   = element_text(size = 2),
      strip.background = element_blank()
    ) +
    labs(y = "*Frequency controlled age z-score", color = label, fill = label) +
    scale_fill_viridis(
      na.value = "#2D004B",
      labels = scales::math_format(10^.x)
    ) +
    scale_colour_viridis(
      na.value = "#2D004B",
      labels = scales::math_format(10^.x)
    )
  if (empty == FALSE) {
    plot <- plot +
      geom_tile()
  }
  return(plot)
}

calculate_psafd_per_variant <- function(df, freq1_col, freq2_col, n1_col, n2_col) {
  df |>
    rowwise() |>  # Perform operations per row (per variant)
    mutate(
      AFD = !!sym(freq1_col) - !!sym(freq2_col),
      PSAFD = AFD / (sqrt(1 / !!sym(n1_col)) + (1 / !!sym(n2_col))),
      AF_log = log10(AF)
      #AFD_log = ifelse(is.na(AFD_log), 0, AFD_log)
    ) |>
    ungroup()
}

generate_af_heatmap_plots <- function(adac_haps, pop_acs_ext, ts_df, thresholds,
                                      AC_thresh = 95,
                                      haplotypes_vec, datasets_vec, datasets_full_names_vec,
                                      AF_col = "AF_log", label = "AF") {
  
  # Step 1: Create hap_rare_list
  hap_rare_list <- list()
  for (haplotype in haplotypes_vec) {
    hap_rare_positions <- adac_haps |>
      filter(mutation_hap == haplotype, AC_ts < AC_thresh) |>
      pull(position)
    
    hap_rare_list[[haplotype]] <- pop_acs_ext |>
      gather_afs(positions_vec = hap_rare_positions, datasets_vec = datasets_vec) |>
      mutate(mutation_hap = haplotype)
  }
  
  # Step 2: Combine data and calculate PS-AFD
  hap_rare_afs <- do.call(rbind, hap_rare_list) |>
    left_join(
      ts_df |>
        dplyr::select(position, AF_ts, time_z) |>
        distinct() |>
        mutate(AN_ts = 95070)
    ) |>
    calculate_psafd_per_variant("AF", "AF_ts", "AN", "AN_ts") |>
    arrange(desc(time_z)) %>%
    mutate(
      position = factor(position, levels = rev(unique(.$position))),
      dataset = case_when(
        dataset == datasets_vec[1] ~ datasets_full_names_vec[1],
        dataset == datasets_vec[2] ~ datasets_full_names_vec[2]
      )
    )
  
  # Step 3: Find thresholds and indices
  hap_rare_afs_uniq <- hap_rare_afs |>
    dplyr::select(position, time_z) |>
    distinct()
  
  results <- lapply(thresholds, function(threshold) {
    filtered_data <- hap_rare_afs_uniq |> filter(time_z < threshold)
    if (nrow(filtered_data) > 0) {
      return(which(hap_rare_afs_uniq$time_z < threshold)[1])
    } else {
      return(NA)
    }
  })
  
  # Step 4: Prepare result vectors
  n <- length(unique(hap_rare_afs$position))
  result_vector <- rep("", n)
  results <- unlist(results)
  results <- results[!is.na(results)]
  
  for (i in 1:length(result_vector)) {
    if (i %in% results) {
      result_vector[i] <- thresholds[which(results == i)]
    }
  }
  
  formatted_result_vector <- sapply(result_vector, function(x) {
    if (x == "") {
      return("")
    } else {
      parse(text = x)
    }
  })
  
  formatted_labels <- function(position) {
    positions_to_label <- which(result_vector != "")
    labels <- rep("", length(position))
  
    # Ensure positions_to_label contains valid indices
    if (length(positions_to_label) > 0) {
      reversed_indices <- rev(seq_along(position)[positions_to_label])
    
      # Ensure reversed_indices only includes valid positions
      valid_indices <- reversed_indices[!is.na(reversed_indices) & reversed_indices <= length(labels)]
    
      # Assign labels only to valid positions
      labels[valid_indices] <- formatted_result_vector[positions_to_label]
    }
  
    labels
  }
  
  # Step 5: Generate heatmap plots
  counter <- 1
  af_heat_plot_list <- list()
  
  for (dataset_name in datasets_full_names_vec) {
    af_heat_plot_list[[counter]] <- hap_rare_afs |>
      filter(dataset == dataset_name) |>
      af_heatmap_plot(AF_col = AF_col, label = label) +
      scale_y_discrete(labels = formatted_labels) +
      theme(plot.margin = unit(c(0, 0.15, 0.15, 0), "lines"))
    
    counter <- counter + 1
  }
  
  # Step 6: Combine plots and add legend
  heatmap_afd <- ggarrange(
    ggarrange(
      af_heat_plot_list[[1]] + theme(legend.position = "NONE"),
      ncol = 1, align = "v", heights = c(0.05, 1, 0.1)
    ),
    ggarrange(
      af_heat_plot_list[[2]] + theme(legend.position = "NONE",
                                     axis.text.y = element_blank(),
                                     axis.title.y = element_blank()),
      ncol = 1, align = "v", heights = c(0.05, 1, 0.1)
    ),
    align = "h", nrow = 1, widths = c(1.5, 1)
  )
  
  legend_afs <- get_legend(af_heat_plot_list[[1]])
  plot_4f <- ggarrange(heatmap_afd, legend_afs, ncol = 2, widths = c(1, 0.3))
  
  return(plot_4f)
}


## FIGURE 5
make_age_cat_boxplot <- function(
  df,
  cat_col,
  categories,
  colors,
  x_var = mean_time,
  title = "Singletons (missense)"
) {
  stopifnot(length(colors) == length(categories))

  # Warn if requested categories aren't present
  present_vals <- unique(na.omit(dplyr::pull(df, {{ cat_col }})))
  missing_levels <- setdiff(categories, present_vals)
  if (length(missing_levels) > 0) {
    warning("These categories aren't present in the data and will be empty: ",
            paste(missing_levels, collapse = ", "))
  }

  df_f <- df %>%
    filter(!is.na({{ cat_col }}), {{ cat_col }} %in% categories) %>%
    mutate(
      {{ cat_col }} := factor({{ cat_col }}, levels = categories),
      .x_log10 = log10({{ x_var }})
    ) %>%
    filter(is.finite(.x_log10))

  if (nrow(df_f) == 0) {
    stop("No data left after filtering (AC_ts==1, category subset, and finite log10).")
  }

  p <- ggplot(
    df_f,
    aes(x = .x_log10, y = {{ cat_col }}, fill = {{ cat_col }}, colour = {{ cat_col }})
  ) +
    geom_boxplot() +
    theme_classic(base_size = 20) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      axis.title = element_blank(),
      legend.position = "none",
      plot.title = element_text(size = rel(0.8))
    ) +
    scale_fill_manual(values = colors, drop = FALSE) +
    scale_colour_manual(values = colors, drop = FALSE) +
    labs(title = title)

  median_df <- df_f %>%
    group_by({{ cat_col }}) %>%
    summarise(median_x = median(.x_log10), .groups = "drop") %>%
    mutate(y_num = as.numeric({{ cat_col }}))

  p +
    geom_point(
      data = median_df,
      aes(x = median_x, y = {{ cat_col }}),
      inherit.aes = FALSE,
      colour = "black"
    ) +
    geom_segment(
      data = median_df,
      aes(
        x = median_x, xend = median_x,
        y = y_num - 0.4, yend = y_num + 0.4
      ),
      inherit.aes = FALSE,
      colour = "black",
      linewidth = 0.6
    )
}

make_age_qq_plot <- function(
  df,
  cat_col,                         # unquoted category column (e.g., AM_pred)
  categories,                      # character vector in desired order
  colors,                          # same length/order as categories
  x_var      = null_times,         # unquoted x (will be log10-transformed in aes)
  y_ratio    = ratio,              # unquoted y line
  y_lower    = lower,              # unquoted ribbon lower
  y_upper    = upper,              # unquoted ribbon upper
  control_level   = NULL,          # e.g., "Likely benign"
  control_suffix  = " (control)",
  xlim_log10 = c(0, 4),
  ylim       = c(0.2, 1.8),
  legend_title = "AlphaMissense class",
  legend_pos   = c(0.05, 0.58),
  title        = NULL
) {
  stopifnot(length(colors) == length(categories))

  # Build display labels (rename only the control level)
  display_labels <- categories
  if (!is.null(control_level) && control_level %in% categories) {
    display_labels[match(control_level, categories)] <- paste0(control_level, control_suffix)
  }

  # Filter to requested categories and set ordered factor with custom labels
  df_f <- df %>%
    filter(!is.na({{ cat_col }}), {{ cat_col }} %in% categories) %>%
    mutate(
      {{ cat_col }} := factor({{ cat_col }}, levels = categories, labels = display_labels)
    )

  if (nrow(df_f) == 0) stop("No data left after filtering to the requested categories.")

  # Map colors to the *display* labels
  color_map <- setNames(colors, levels(dplyr::pull(df_f, {{ cat_col }})))

  ggplot(
    df_f,
    aes(
      x     = log10({{ x_var }}),
      y     = {{ y_ratio }},
      color = {{ cat_col }},
      fill  = {{ cat_col }}
    )
  ) +
    geom_line() +
    geom_ribbon(aes(ymin = {{ y_lower }}, ymax = {{ y_upper }}), alpha = 0.4, linewidth = 0.05) +
    labs(
      color = legend_title,
      fill  = legend_title,
      x = "Control quantile allele age (generations)",
      y = "Category / control quantile age ratio",
      title = title
    ) +
    theme_bw(base_size = 20) +
    theme(
      plot.title = element_text(size = rel(0.8)),
      axis.text  = element_text(size = rel(0.8)),
      panel.grid = element_blank(),
      legend.position = legend_pos,
      legend.justification = c(0, 0),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white")
    ) +
    coord_cartesian(xlim = xlim_log10, ylim = ylim) +
    scale_x_continuous(expand = c(0, 0), labels = scales::math_format(10^.x)) +
    scale_y_continuous(expand = c(0, 0)) +
    annotation_logticks(
      sides = "b",
      color = "grey40",
      short = unit(0.05, "cm"),
      mid   = unit(0.075, "cm"),
      long  = unit(0.1, "cm")
    ) +
    scale_fill_manual(values = color_map, guide = guide_legend(override.aes = list(alpha = 1))) +
    scale_colour_manual(values = color_map, guide = guide_legend(override.aes = list(colour = "white")))
}

cat_n_plot <- function(
  df,
  cat_col,                       # unquoted, e.g. AM_pred
  categories,                    # character vector in desired order
  colors,                        # same length/order as categories
  # Filtering helpers
  filter_ac_ts   = TRUE,         # apply AC_ts <= max_ac_fun(df, AF_thresh)
  AF_thresh      = 0.001,
  max_ac_fun     = max_ac_count, # function(df, AF_thresh) -> max AC
  require_non_na = NULL,         # unquoted col to drop NA from (e.g. AM_score)
  # Counting
  weight         = NULL,         # unquoted numeric col to sum instead of n()
  # Presentation
  x_divisor      = 1000,         # divide counts for axis scaling (e.g. 1000 -> 'thousands')
  x_label        = "Variants (thousands)",
  title          = NULL,
  legend_title   = NULL,         # e.g. "AlphaMissense class"
  show_legend    = FALSE         # match your original (no legend)
) {
  stopifnot(length(colors) == length(categories))

  # Start
  df_f <- df

  # Optional AC_ts filter using provided function + threshold
  if (isTRUE(filter_ac_ts)) {
    max_ac <- max_ac_fun(df_f, AF_thresh = AF_thresh)
    df_f <- df_f %>% filter(AC_ts <= max_ac)
  }

  # Optional NA drop on a given column
  if (!quo_is_null(enquo(require_non_na))) {
    df_f <- df_f %>% tidyr::drop_na({{ require_non_na }})
  }

  # Filter to categories & order factor
  df_f <- df_f %>%
    filter(!is.na({{ cat_col }}), {{ cat_col }} %in% categories) %>%
    mutate({{ cat_col }} := factor({{ cat_col }}, levels = categories))

  # Tally or weighted sum
  if (quo_is_null(enquo(weight))) {
    agg <- df_f %>%
      group_by({{ cat_col }}) %>%
      tally(name = "n") %>%
      ungroup()
  } else {
    agg <- df_f %>%
      group_by({{ cat_col }}) %>%
      summarise(n = sum({{ weight }}, na.rm = TRUE), .groups = "drop")
  }

  if (nrow(agg) == 0) stop("No data to plot after filtering/grouping.")

  # Single-row stacked bar: constant y; scaled x
  agg <- agg %>%
    mutate(
      y_const = "y",
      x_scaled = n / x_divisor
    )

  # Colour map aligned to factor levels
  level_names <- levels(agg %>% pull({{ cat_col }}))
  color_map <- setNames(colors, level_names)

  p <- ggplot(
    agg,
    aes(x = x_scaled, y = y_const, fill = {{ cat_col }}, colour = {{ cat_col }})
  ) +
    geom_bar(stat = "identity") +
    theme_classic(base_size = 20) +
    theme(
      axis.ticks.y = element_blank(),
      axis.text.y  = element_blank(),
      axis.line.y  = element_blank(),
      legend.position = if (show_legend) "right" else "none",
      plot.title   = element_text(size = rel(0.8))
    ) +
    labs(
      x = x_label,
      y = "",
      title = title,
      fill  = legend_title,
      colour = legend_title
    ) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = color_map) +
    scale_colour_manual(values = color_map)

  p
}

enrichment_depletion_plot <- function(
  log_df,
  group_col = "pop",
  group_vals = NULL,
  outlier_scores = c(1, 3, 10, 30),
  log10_outlier_label = FALSE,
  coord_limits = c(-0.75, 0.75),
  colours_vec = c("#238b45", "#74c476", "grey80"),
  plot_title = "GEL DAF < 0.1% (missense)"
) {
  group_sym <- rlang::sym(group_col)

  # Filter and convert the grouping variable to a factor if group_vals is specified
  if (!is.null(group_vals)) {
    log_df %<>%
      filter({{ group_sym }} %in% group_vals) %>%
      mutate(
        {{ group_sym }} := factor({{ group_sym }}, levels = group_vals)
      )
  }

  # Filter the dataframe for the specified outlier scores and create log-transformed labels
  log_df %<>%
    filter(outlier_score %in% outlier_scores)

  if(log10_outlier_label) {
    log_df %<>%
      mutate(
        log_score = round(log10(outlier_score), 1),
        outlier_score_label = paste0("10^", log_score),
        outlier_score_label = factor(outlier_score_label, levels = paste0("10^", round(log10(outlier_scores), 1)))
      )
  } else {
    log_df %<>%
      mutate(
        outlier_score_label = outlier_score,
        outlier_score_label = factor(outlier_score_label, levels = outlier_scores)
      )
  }
  
  log_df %>%
    ggplot(aes(
      x = {{ group_sym }},
      y = estimate,
      fill = {{ group_sym }}
    )) +
    geom_bar(stat = "identity", colour = "white", position = "dodge") +
    geom_hline(yintercept = 0, colour = "grey60") +
    geom_errorbar(
      aes(ymax = conf.high, ymin = conf.low),
      position = position_dodge(width = 0.9),
      colour = "grey30",
      width = 0.3
    ) +
    facet_wrap(
      ~outlier_score_label,
      nrow = 1,
      strip.position = "bottom"
    ) +
    theme_bw(base_size = 20) +
    theme(
      strip.background = element_blank(),
      legend.position = "none",
      axis.text.x = element_blank(),
      axis.line.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid = element_blank(),
      strip.text = element_text(hjust = 0.5, margin = margin(t = 5, b = 5))
    ) +
    labs(x = NULL, y = "Estimate") +
    theme(plot.title = element_text(size = rel(0.8))) +
    coord_cartesian(ylim = coord_limits) +
    scale_fill_manual(
      values = colours_vec
    ) +
    labs(
      x = "Frequency controlled age z-score bin",
      title = plot_title,
      y = "Relative log P(class membership)"
    ) 
}


### FIGURE X
gather_afs <- function(
  acs_df,
  positions_vec,
  exclude_pops = c("remaining"),
  datasets_vec
) {
  acs_df %>%
  filter(position %in% positions_vec) %>%
    pivot_longer(
      !c(position),
      names_to = "name",
      values_to = "value"
    ) %>%
    mutate(
      ac_an = case_when(
        grepl("AN", name) ~ "AN",
        grepl("AC", name) ~ "AC"
      ),
      name = gsub("AN_", "", gsub("AC_", "", name))
    ) %>%
    drop_na(ac_an) %>%
    pivot_wider(
      names_from = "ac_an",
      values_from = "value"
    ) %>%
    mutate(
      AF = AC / AN,
      AF = ifelse(is.na(AF), 0, AF)
    ) %>%
    separate(
      name,
      sep = "_",
      into = c("pop", "dataset", "type")
    ) %>%
    filter(
      !(pop %in% exclude_pops),
      dataset %in% datasets_vec
    ) %>%
    drop_na(type) %>%
    mutate(
      AF_log = ifelse(log10(AF) == -Inf, NA, log10(AF))
    )
}

generate_gnomad_afs_data <- function(data_dir, example_mut, region_length, pops_df, group_numbers) {
  # Read in the data
  column_names <- names(read_tsv(file.path(paste0(data_dir, "all-", example_mut$chromosome, "-df.tsv.gz")), n_max = 1))
  selected_columns <- column_names[grepl("position|gnomad_exomes", column_names)]
  
  # Gather AFS data
  pop_afs_ext <- read_tsv(
    paste0(data_dir, "all-", example_mut$chromosome, "-df.tsv.gz"),
    col_select = all_of(selected_columns)
  ) %>%
  filter(position == example_mut$position) %>%
  dplyr::select(-contains(c("AF", "AN_gnomad_exomes", "AC_gnomad_exomes"))) %>%
  gather_afs(
    positions_vec = example_mut$position,
    exclude_pops = "none",
    datasets_vec = "gnomad"
  ) %>%
  mutate(
    pop = factor(pop, levels = rev(group_numbers$pop))
  )
  
  return(pop_afs_ext)
}

# Function for gnomAD AFS Plot
generate_gnomad_afs_plot <- function(pop_afs_ext, pop_colours_vec) {
  gnomad_afs_plot <- pop_afs_ext %>%
    ggplot(aes(
      x = pop, y = AF, fill = pop, colour = pop
    )) +
    geom_bar(stat = "identity") +
    geom_text(aes(
      label = ifelse(AF == 0, NA, paste0(round(AF * 100, 3), "%"))
    ), vjust = -0.5, size = 2) +
    theme_void() +
    scale_fill_manual(values = rev(pop_colours_vec)) +
    scale_colour_manual(values = rev(pop_colours_vec)) +
    scale_y_continuous(limits = c(0, 0.002)) +
    theme(
      plot.margin = unit(c(0, 0, 0, 0), "lines"),
      legend.position = "NONE",
      plot.title = element_text(hjust = 1, size = 10, colour = "grey20")
    ) +
    labs(title = "gnomADv4.1 (exomes) AFs  ")

  return(gnomad_afs_plot)
}

generate_example_mut_pops <- function(example_mut, pops_df, ts_dir, trees_list, region_length) {
  # Extract ts_name from example_mut and filter corresponding tree
  ts_name = example_mut$ts_name
  tree <- trees_list %>% filter(grepl(ts_name, ts_prefix))
  
  # Get the carrier IDs for the given mutation position
  example_mut_ids <- get_carriers_ids(
    ts_path = paste0(ts_dir, tree$ts_prefix, ".trees.tsz"),
    positions = c(example_mut$position, 10000000000) # fake position for terrible iterable code necessity
  )
  
  # Create the example_mut_pops dataframe with the necessary details
  example_mut_pops <- pops_df %>%
    filter(sgkit_sample_id %in% example_mut_ids$carrier_id) %>%
    mutate(
      mean_time = 20000,
      position = example_mut$position + region_length - (5000 * nrow(.)) + (row_number() - 1) * 5000
    )
  example_mut_pops %>%
    pull(sgkit_sample_id) %>%
    unique() %>%
    length() -> n_carriers
  if (n_carriers < example_mut$AC_ts) {
    print("AT LEAST ONE CARRIER IS HOMOZYGOUS FOR THIS MUTATION\n")
  }
  
  return(example_mut_pops)
}

# Function for main plot (to_show_plot)
generate_to_show_plot <- function(
  ts_df_merge, example_mut, region_length,
  example_mut_pops, group_numbers, annot_line
  ) {
  to_show_plot <- ts_df_merge %>%
    filter(
      chromosome == unique(example_mut$chromosome) &
        position > example_mut$position - region_length &
        position < example_mut$position + region_length &
        AC_ts == example_mut$AC_ts
    ) %>%
    arrange(position) %>%
    mutate(log10_mean_time = log10(mean_time)) %>%
    filter(!is.na(log10_mean_time)) %>%
    mutate(
      rolling_avg = rollmean(log10(mean_time), 100, fill = NA, align = "center")
    ) %>%
    ggplot(aes(x = position, y = log10(mean_time))) +
    geom_vline(
      xintercept = example_mut$position,
      colour = "grey80", linewidth = 0.075
    ) +
    geom_line(aes(x = position, y = rolling_avg), color = "grey40", size = 0.25) +
    geom_hline(
      linewidth = 0.075,
      yintercept = log10(annot_line),
      colour = "grey80"
    ) +
    geom_rect(
      data = example_mut,  
      aes(
        xmin = edge_left,
        xmax = edge_right,
        ymin = log10(child_node_mean_time),
        ymax = log10(parent_node_mean_time)
      ),
      inherit.aes = FALSE,
      fill = "#69B3E7",
      alpha = 0.25
    ) +
    geom_rect(
      data = example_mut,  
      aes(
        xmin = ibd_left,
        xmax = ibd_right,
        ymin = log10(child_node_mean_time),
        ymax = log10(ibd_time)
      ),
      inherit.aes = FALSE,
      fill = "#EBCC2A",
      alpha = 0.25
    ) +
    geom_segment(
      data = example_mut,  
      aes(
        x = ibd_left,
        xend = ibd_right,
        y = log10(ibd_time),
        yend = log10(ibd_time)
      ),
      inherit.aes = FALSE,
      linetype = "dotted",
      colour = "grey20"
    ) +
    geom_point(
      data = example_mut_pops,
      aes(x = position, y = log10(mean_time), fill = pop, colour = "white"),
      shape = 22, size = 4
    ) +
    geom_point(
      data = example_mut, aes(x = position, y = log10(mean_time)),
      color = "#69B3E7"
    ) +
    geom_errorbar(
      data = example_mut, aes(x = position, ymax = log10(child_node_mean_time), ymin = log10(parent_node_mean_time)),
      color = "#69B3E7"
    ) +
    theme_bw() +
    theme(
      legend.position = "NONE",
      panel.grid = element_blank()
    ) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(
      expand = c(0,0),
      labels = scales::math_format(10^.x)
    ) +
    annotation_logticks(
      sides = "l",
      color = "grey40",
      short = unit(0.05, "cm"),
      mid = unit(0.075, "cm"),
      long = unit(0.1, "cm")
    ) +
    coord_cartesian(
      ylim = c(0, 5),
      xlim = c(example_mut$position - region_length, example_mut$position + region_length)
    ) +
    scale_fill_manual(values = setNames(group_numbers$colour, group_numbers$pop)) +
    scale_colour_manual(values = setNames(group_numbers$colour, group_numbers$pop)) +
    labs(
      x = paste0("Chromosome ", gsub("chr", "", example_mut$chromosome)),
      y = "Age (generations ago)"
    )

  return(to_show_plot)
}

generate_to_show_plot_annots <- function(
  example_mut, example_mut_pops, to_show_plot, annot_line = 2000, annot_text = "Out-of-Africa migration"
) {
  
  if (example_mut$ibd_left > (example_mut$position - region_length)) {
    ibd_annot_x = example_mut$ibd_left
  } else {
    ibd_annot_x = example_mut$position - (region_length - 2500)
  }

  to_show_plot <- to_show_plot +
    annotate(
      "text",
      x = example_mut$position - (region_length - 2250),
      y = log10(annot_line) + .2,
      label = annot_text,
      hjust = 0,
      size = 3,
      colour = "grey20"
    ) +
   annotate(
      "text",
      x = ibd_annot_x,
      y = log10(example_mut$ibd_time) + .2,
      label = paste0("IBD span: ", round(example_mut$cM_length, 2), "cM"),
      hjust = 0,
      size = 3,
      colour = "grey20"
    ) +
    annotate(
      "text",
      x = min(example_mut_pops$position) - 5000,
      y = log10(unique(example_mut_pops$mean_time)),
      hjust = 1,
      label = "Carrier PCA group(s)",
      colour = "grey20", size = 3
    ) +
    annotate(
      "text",
      x = max(example_mut_pops$position) - 2250,
      y = log10(50000),
      label = paste0("AC: ", example_mut$AC_ts),
      colour = "grey20", size = 3
    ) +
    annotate(
     "text",
      x = example_mut$position - 10000,
      y = log10(example_mut$mean_time),
      label = "Mutation",
      colour = "grey20", size = 3
    ) +
    annotate(
      "text",
      x = example_mut$edge_right + 2000,
      y = log10(example_mut$mean_time),
      label = "Edge",
      angle = 90,
      colour = "grey20", size = 3
    )

    return(to_show_plot)
}

generate_base_plot_data <- function(ensembl_transcripts, example_mut, region_length, max_genes = 6) {
  # Filter and annotate genes and exons
  ensembl_example <- ensembl_transcripts %>%
    filter(
      gene_name == example_mut$gene_symbol |
      (chromosome ==  example_mut$chromosome &
       start > example_mut$position - region_length &
       end < example_mut$position + region_length)
    ) %>%
    mutate(
      green_gene = ifelse(gene_name == example_mut$gene_symbol, TRUE, FALSE)
    )
  
  genes <- ensembl_example %>%
    filter(type == "gene") %>%
    filter(gene_biotype == "protein_coding")
  
  exons <- ensembl_example %>%
    filter(type == "exon" & gene_name %in% genes$gene_name & tag %in% c("Ensembl_canonical", "MANE_Select"))
  
  # Make sure we have a fixed number of genes to plot, if there are fewer than max_genes, add empty ones
  if (nrow(genes) < max_genes) {
    missing_genes <- max_genes - nrow(genes)
    empty_genes <- tibble(
      gene_name = rep(NA, missing_genes),
      y_position = rep(NA, missing_genes),
      green_gene = rep(FALSE, missing_genes)
    )
    genes <- bind_rows(genes, empty_genes)
  }
  
  genes$y_position <- seq(0, by = 2, length.out = nrow(genes))
  genes <- genes %>%
    arrange(desc(green_gene), y_position)
  
  genes$y_position <- seq(0, by = 2, length.out = nrow(genes))
  exons$y_position <- genes$y_position[match(exons$gene_name, genes$gene_name)]
  
  exon_example <- exons %>% filter(start <= example_mut$position & end >= example_mut$position)
  
  return(list(genes = genes, exons = exons, exon_example = exon_example))
}

#238b45
generate_base_plot <- function(genes, exons, example_mut, region_length, gene_colour) {
  base_plot <- ggplot() +
    geom_rect(data = exons,
              aes(xmin = start, xmax = end, ymin = y_position - 0.5, ymax = y_position + 0.5),
              fill = ifelse(exons$green_gene,  gene_colour, "grey70")) +
    geom_segment(data = genes, 
                 aes(x = start, xend = end, y = y_position, yend = y_position),
                 colour = ifelse(genes$green_gene, gene_colour , "grey70"), 
                 size = 0.45, linetype = "dotted") +
    geom_text(data = genes %>% filter(green_gene == TRUE), 
              aes(x = end, y = y_position, label = gene_name), 
              color =  gene_colour , fontface = "bold", size = 3, 
              hjust = 1, vjust = -2) +  # Rotate text and position it to the left
  
    labs(x = paste0("Chromosome ", gsub("chr", "", example_mut$chromosome)), y = "") +
    theme_bw() +
    theme(legend.position = "NONE",
          panel.grid = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank()) + 
    coord_cartesian(
      xlim = c(example_mut$position - region_length, example_mut$position + region_length),
      ylim = c(min(genes$y_position) - 1, max(genes$y_position) + 1)  # Set dynamic Y-axis limits based on genes
    ) + 
  theme(
    axis.text.x = element_blank(),  # Remove x axis labels for cleaner overlay
    axis.ticks.x = element_blank(), # Remove x axis ticks for cleaner overlay
    axis.title.x = element_blank(), # Remove x axis title for cleaner overlay
    plot.background = element_blank(),  # Remove the plot background
    panel.background = element_blank(), # Remove the panel background
    panel.border = element_blank(),  # Remove the panel border
  )
  
  return(base_plot)
}

combine_plots_with_data <- function(
  data_dir, example_df, region_length, ts_df_merge, pops_df,
  group_numbers, ensembl_transcripts, mutation_title, ts_dir, trees_list,
  annot_line = 2000, annot_text = "Out-of-Africa migration", gene_colour = "#238b45"
) {
  
  example_mut <- example_df %>%
    mutate(
      ibd_left = ifelse(
        ibd_left > .$position - region_length,
        ibd_left, 
        .$position - region_length
      ),
      ibd_right = ifelse(
        ibd_right < .$position + region_length,
        ibd_right, 
        .$position + region_length
      )
    )

  # Generate necessary data
  print("Generating plotting data...")
  example_mut_pops <- generate_example_mut_pops(example_mut, pops_df, ts_dir, trees_list, region_length)
  pop_afs_ext <- generate_gnomad_afs_data(data_dir, example_mut, region_length, pops_df, group_numbers)
  base_plot_data <- generate_base_plot_data(ensembl_transcripts, example_mut, region_length)
  
  # Generate the plots
  print("Generating plots...")
  to_show_plot <- generate_to_show_plot(ts_df_merge, example_mut, region_length, example_mut_pops, group_numbers, annot_line)
  to_show_plot_annot <- generate_to_show_plot_annots(example_mut, example_mut_pops, to_show_plot, annot_line, annot_text)
  gnomad_afs_plot <- generate_gnomad_afs_plot(pop_afs_ext, group_numbers$colour)
  base_plot <- generate_base_plot(base_plot_data$genes, base_plot_data$exons, example_mut, region_length, gene_colour)
  
  # Generate the title grob
  title_grob <- textGrob(mutation_title, gp = gpar(fontsize = 10, colour = "grey20"))
  aligned_plots <- align_plots(to_show_plot_annot, base_plot, align="hv", axis="tblr")
  main_plot <- ggdraw(aligned_plots[[1]]) + draw_plot(aligned_plots[[2]])
  
  # Arrange the plots and combine them
  print("Combining the plots...")
  comb_plot <- ggarrange(
    ggarrange(ggarrange(NULL, title_grob, ncol = 1, heights = c(1, 0.2)), NULL, gnomad_afs_plot, nrow = 1, widths = c(0.45, 0.675, 1)),
    main_plot,
    ncol = 1,
    align = "v",
    heights = c(0.4, 1.1)
  )
  
  return(comb_plot)
}

# helper: colour mapping with one highlighted
col_vec_for <- function(target_pop, other_colour = "grey85") {
  cv <- stats::setNames(rep(other_colour, length(pop_cols)), names(pop_cols))
  cv[target_pop] <- pop_cols[[target_pop]]
  cv
}
# function returning the ggplot object
plot_one_pop_classify <- function(target_pop, AC = 2) {
  df <- annot_df %>%
    filter(recurrent_unlikely & AC_ts == AC) %>%
    arrange(desc(mean_time)) %>%
    mutate(
      position = factor(position, levels = .$position),
      pop = factor(pop, levels = pop_vec)
    )

  col_vals <- col_vec_for(target_pop)

  ggplot(df, aes(x = position, y = mean_time, colour = pop)) +
    geom_errorbar(
      aes(ymax = parent_node_mean_time, ymin = child_node_mean_time,
          alpha = pop == target_pop),
      linewidth = 0.5
    ) +
    geom_point(aes(alpha = pop == target_pop), size = 0.5) +
    theme_classic(base_size = 25) +
    scale_color_manual(values = col_vals, guide = "none") +
    scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.15), guide = "none") +
    theme(
      axis.text.x = element_blank(),
      strip.background = element_blank(),
      axis.ticks.x = element_blank()
      #legend.position = "none"
    ) +
    scale_y_log10(
      expand = c(0,0),
      breaks = scales::trans_breaks("log10", function(z) 10^z),
      labels = scales::trans_format("log10", scales::math_format(10^.x))
    ) + 
    labs(
      title = target_pop,
      x = "",
      y = "Allele age (generations)"
    ) +
    facet_grid(~acmg_classification_broad, scales = "free_x", space = "free_x")
}


plot_obs_vs_random <- function(random_values, observed_value,
                               title = "Observed vs Random",
                               xlab  = "Statistic") {

  df <- data.frame(random = random_values)

  ggplot(df, aes(x = random)) +
    geom_histogram(aes(y = ..density..), bins = 40,
                   colour = "grey30", fill = "steelblue", alpha = 0.65) +
    geom_density(colour = "navy", linewidth = 1) +
    geom_vline(xintercept = observed_value,
               colour = "red", linewidth = 1.2) +
    annotate("text", x = observed_value, y = Inf,
             label = sprintf("Observed = %.3f", observed_value),
             vjust = -0.5, hjust = -0.1, colour = "red", fontface = "bold") +
    labs(
      title = title,
      x = xlab,
      y = "Density"
    ) +
    theme_minimal(base_size = 14)
}