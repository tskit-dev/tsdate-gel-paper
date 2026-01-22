#!/usr/bin/env Rscript

### Function for getting date info from each tree sequence + combining them. 
combine_ts_dfs <- function(
  chrom,
  ts_list,
  df_dir,
  df_suffix = "-times.tsv",
  ts_dir,
  ts_suffix = ".trees.tsz",
  pops_path = NULL
  ) {
  
  # List of trees within this chromosome
  trees_list <- read_tsv(ts_list)
  trees_list_chrom <- trees_list %>%
    filter(chromosome == chrom) %>%
    mutate(
      df_files = paste0(ts_prefix, df_suffix),
      ts_files = paste0(ts_prefix, ts_suffix)
    )
  # Combine
  df_list <- list()
  for (index in 1:nrow(trees_list_chrom)) {
    df_file <- paste0(
      df_dir,
      trees_list_chrom$df_files[index]
    )
    if (file.exists(df_file)) {
      df_list[[index]] <- read_tsv(
        file.path(df_file),
        lazy = TRUE
      )
    } else {
      cat(paste0("Data frame: ", df_file, " does not exist.\nCreating from tree sequence.\n"))
      ts_path <- paste0(ts_dir, trees_list_chrom$ts_files[index])
      
      if (df_suffix == "-times.tsv") {
        cat(paste0(
          "Creating data frame dumping dates from: ",
          ts_path, "\n"
        ))
        dates_df <- dump_times_data_frame(ts_path = ts_path) %>%
          mutate(
            chromosome = chrom,
            ts_name = strsplit(trees_list_chrom$ts_prefix[index], "-")[[1]][2]
          )
        ## Write to file
        write.table(
          dates_df,
          df_file,
          row.names = FALSE,
          col.names = TRUE,
          quote = FALSE,
          sep = "\t"
        )
        df_list[[index]] <- dates_df

      } else if (df_suffix == "-pop_counts.tsv") {
        cat(paste0(
          "Creating data frame dumping dates from: ",
          ts_path,
          " using pop assignments from: ",
          pops_path, "\n"
        ))
        pop_counts_df <- dump_pop_allele_counts_data_frame(
          ts_path = ts_path,
          pops_path = pops_path
        )
        write.table(
          pop_counts_df,
          df_file,
          row.names = FALSE,
          col.names = TRUE,
          quote = FALSE, sep = "\t"
        )
        df_list[[index]] <- pop_counts_df
      }
    }
  }
  out_frame <- do.call(rbind, df_list) %>%
    arrange(desc(position))
}

## where time_var = posterior mean allele age
add_date_ac_stats <- function(df, time_col = "mean_time", var_col = "var_time") {
  time_col_sym <- rlang::sym(time_col)
  var_col_sym <- rlang::sym(var_col)
  df %>%
    mutate(
      position = as.integer(position),
      shape_time = {{time_col_sym}}^2 / {{var_col_sym}},
      scale_time = {{var_col_sym}} / {{time_col_sym}},
      sd_time = sqrt({{var_col_sym}}),
      cv_time = sd_time / {{time_col_sym}} * 100,
      lower_conf_time = qgamma(0.025, shape = shape_time, scale = scale_time),
      upper_conf_time = qgamma(0.975, shape = shape_time, scale = scale_time)
    )
}

pop_specific_alleles <- function(
  df,
  ac_col = "AC_ts",
  pop_col_contains = "ts",
  colname = "pop",
  remain_collapse = NULL
) {

  # Syms
  ac_col_sym <- rlang::sym(ac_col)
  col_name_sym <- rlang::sym(colname)
  
  # Dynamically identify population-specific columns
  pop_cols <- names(df) %>%
    grep(pattern = paste0("^AC_.*_", pop_col_contains), value = TRUE)
  
  # Add a new column dynamically based on population columns
  df_restricted <- df %>%
    dplyr::select(
      chromosome,
      position,
      {{ac_col_sym}},
      all_of(pop_cols)
    ) %>%
    rowwise() %>%
    mutate(
      {{col_name_sym}} := {
        # Iterate over population columns and find the matching population
        pop_match <- purrr::map_chr(pop_cols, function(pop_col) {
          if (get(pop_col) == !!ac_col_sym) {
            # Extract the population name (remove "AC_" and "_ts")
            pop_name <- sub("^AC_", "", pop_col)
            sub("_ts$", "", pop_name)  
          } else {
            NA_character_
          }
        })
        pop_match <- pop_match[!is.na(pop_match)]  # Remove NA matches
        if (length(pop_match) == 1) {
          pop_match  # Return the matching population
        } else {
          "nonspecific"
        }
      }
    ) %>%
    ungroup() %>%
    mutate(
      # Remove non-specific singletons (shouldn't happen if all inds have pop counts)
      {{col_name_sym}} := ifelse(
        {{col_name_sym}} == "nonspecific" & {{ac_col_sym}} == 1,
        NA_character_,
        {{col_name_sym}}
      ),
      within_across = ifelse(
        is.na({{col_name_sym}}),
        "Across populations",
        "Within population"
      )
    ) %>%
    dplyr::select(
      chromosome,
      position,
      {{col_name_sym}},
      within_across
    )

  if(!is.null(remain_collapse)) {
    df_restricted %<>%
      mutate(
        {{col_name_sym}} := ifelse(
          {{col_name_sym}} %in% remain_collapse,
          "remaining",
          {{col_name_sym}}
        )
      )
  }
  
  # Merge the results back to the original dataframe
  df %>%
    left_join(
      df_restricted,
      by = c("chromosome", "position")
    )
}

filt_to_ts_align_ancestor <- function(
  data,
  ts_alleles,
  chr_pos_ref_alt = c(
    "chromosome",
    "position",
    "reference",
    "alternate"
  ),
  ac_col = "AC_",
  an_col = "AN_",
  flip_acs = FALSE,
  exclude_derived_refs = FALSE,
  remove_reference_is = TRUE
  ) {
  
  if (length(chr_pos_ref_alt) != 4) {
    stop(paste0(
      "Argument chr_pos_ref_alt: ",
      chr_pos_ref_alt,
      " should be a vector with 4 column names.\n"
    ))
  }
  
  ## Syms
  chr_sym <- rlang::sym(chr_pos_ref_alt[1])
  pos_sym <- rlang::sym(chr_pos_ref_alt[2])
  ref_sym <- rlang::sym(chr_pos_ref_alt[3])
  alt_sym <- rlang::sym(chr_pos_ref_alt[4])

  ## Filter on positions in the ts
  data_filt <- data %>%
    dplyr::rename(
      chromosome = {{chr_sym}},
      position = {{pos_sym}},
      reference = {{ref_sym}},
      alternate = {{alt_sym}}
    ) %>%
    mutate(position = as.integer(position)) %>%
    filter(position %in% ts_alleles$position) %>%
    left_join(
      ts_alleles %>% dplyr::select(
        chromosome,
        position,
        ancestral_state,
        derived_state
      ),
      by = c("chromosome", "position")) %>%
    filter(
      (alternate == ancestral_state & reference == derived_state) | 
      (alternate == derived_state & reference == ancestral_state)
    )

  # Reference is the derived state (polarise ACs)
  data_der_ref <- data_filt %>%
    filter(reference == derived_state) %>%
    mutate(reference_is = "derived_state")
  
  # If the data contains allele counts...
  # ... flip these so they are aligned with anc/der encoding.
  if (flip_acs) {
    data_der_ref <- data_der_ref %>%
      mutate((across(contains(ac_col)) - across(contains(an_col))) * -1)
  }

  # Reference is ancestral state
  data_anc_ref <- data_filt %>%
    filter(reference == ancestral_state) %>%
    mutate(reference_is = "ancestral_state")

  # Create the 
  data_aligned <- rbind(
    data_der_ref,
    data_anc_ref
  ) %>%
  dplyr::select(-c(reference, alternate))
  
  if (exclude_derived_refs) {
    data_aligned %<>%
      filter(reference_is == "ancestral_state")
  }

  if (remove_reference_is) {
    data_aligned %<>%
      dplyr::select(-reference_is)
  }
  return(data_aligned)
}

##### Function for pulling VEP chunked TSVs
annotate_vep_chunks <- function(
  chunks_tsv,
  vep_dir,
  vep_suffix,
  consequence_vector = NULL,
  transcript_ids_vector = NULL
) {

  vep_chunk_list <- vector("list", nrow(chunks_tsv))
  for (chunk_index in 1:nrow(chunks_tsv)) {
    chromosome <- chunks_tsv[chunk_index,] %>%
      pull(chromosome)
    start <- chunks_tsv[chunk_index,] %>%
      pull(start)
    end <- chunks_tsv[chunk_index,] %>%
      pull(end)
    cat(paste0(
      "Pulling VEP109 consequence info for chunk. ",
      chromosome,
      " start: ", start,
      " end: ", end, "\n")
    )
    ## Add VEP annotation to chunk
    vep_chunk_list[[chunk_index]] 
    vep_chunk <- 
      fread(paste0(
        vep_dir,
        chromosome, "_",
        start, "_",
        end, 
        vep_suffix)) %>%
      distinct()
        
    if(!is.null(consequence_vector)) {
      vep_chunk %<>%
        filter(
          grepl(paste(consequence_vector, collapse = "|"),
          consequence)
        )
    }
    vep_chunk_list[[chunk_index]] <- vep_chunk
  }
  vep_annots <- do.call(rbind, vep_chunk_list) %>%
    distinct()

  ## Specify list of ENSEMBL gene ids 
  if(!is.null(transcript_ids_vector)) {
    vep_annots <- vep_annots %>%
      filter(transcript_id %in% transcript_ids_vector) %>%
      distinct()
  }
  return(vep_annots)
}

encode_archaic_genotypes <- function(
  archaic_alts_df,
  genotype_col,
  ts_alleles_ref_is,
  filter_col,
  filter_alleles = NULL
) {
  genotype_sym <- rlang::sym(genotype_col)
  out_df <- ts_alleles_ref_is %>%
    left_join(archaic_alts_df) %>%
    filter({{genotype_sym}} %in% c("0/1", "1/1") | is.na({{genotype_sym}})) %>%
    mutate(
      {{genotype_sym}} := case_when(
        reference_is == "derived_state" & {{genotype_sym}} == "1/1" ~ "0/0",
        reference_is == "derived_state" & is.na({{genotype_sym}}) ~ "1/1",
        reference_is == "ancestral_state" & is.na({{genotype_sym}}) ~ "0/0",
        TRUE ~ {{genotype_sym}}
      )
    )
  
  filter_sym <- rlang::sym(filter_col)
  if(!is.null(filter_alleles)) {
    out_df %<>%
      filter({{filter_sym}} %in% filter_alleles)
  }
  out_df %<>% 
    dplyr::select(-c({{filter_sym}}, reference_is))
  return(out_df)
}

#### LabKey functions
sqlquery <- function(sql) {
  labkey.executeSql(
    sql = sql,
    maxRows = 1000000000,
    folderPath = labkey,
    schemaName = "lists",
    colNameOpt = "rname")
}

clinvar_to_exit_encode_acmg <- function(
  clinvar_df,
  clin_sig_col = "clinvar_significance"
  ) {
  clin_sig_sym <- rlang::sym(clin_sig_col)
  clinvar_df %>%
    filter({{clin_sig_sym}} != "not_provided") %>%
    mutate(acmg_classification = case_when(
      {{clin_sig_sym}} == "association" ~ "association_variant",
      {{clin_sig_sym}} == "Benign" ~ "benign_variant",
      {{clin_sig_sym}} %in% c("Likely benign", "Benign/Likely benign") ~ "likely_benign_variant",
      {{clin_sig_sym}} == "Pathogenic" ~ "pathogenic_variant",
      {{clin_sig_sym}} %in% c("Likely pathogenic", "Pathogenic/Likely pathogenic") ~ "likely_pathogenic_variant",
      grepl("Conflicting", {{clin_sig_sym}}) ~ "conflicting_variant",
      {{clin_sig_sym}} == "Uncertain significance" ~ "variant_of_unknown_clinical_significance"
    )) %>%
     dplyr::select(-{{clin_sig_sym}})
}

pull_exit_100k <- function(sql = NULL, labkey) {
  if (is.null(sql)) {
    sql <- paste0("SELECT
      ex.chromosome,
      ex.acmg_classification,
      ex.position,
      ex.reference,
      ex.alternate,
      FROM gmc_exit_questionnaire as ex
      WHERE ex.assembly = 'GRCh38'")
  }
  out <- sqlquery(sql) %>%
    mutate(
      reference = na_if(reference, "NA"),
      alternate = na_if(alternate, "NA"),
      chromosome = paste0("chr", gsub("chr", "", chromosome)),
    ) %>%
    distinct() %>%
    drop_na(chromosome)
  return(out)
}

pull_diagnostic_discovery <- function(sql = NULL, labkey) {
  if (is.null(sql)) {
    sql <- paste0("SELECT
      dd.chromosome,
      dd.position,
      dd.reference,
      dd.alternate,
      FROM submitted_diagnostic_discovery as dd
      WHERE dd.genome_build = 'GRCh38'")
  }
  out <- sqlquery(sql) %>%
    mutate(
      reference = na_if(reference, ""),
      alternate = na_if(alternate, ""),
      chromosome = paste0("chr", gsub("chr", "", chromosome)),
      acmg_classification = "likely_pathogenic_variant"
    ) %>%
    distinct()
  return(out)
}

pull_exit_gms <- function(sql = NULL, labkey) {
  if (is.null(sql)) {
    sql <- paste0("SELECT
      ex.chromosome,
      ex.acmg_classification,
      ex.position,
      ex.reference,
      ex.alternate,
      FROM report_outcome_questionnaire as ex
      WHERE ex.genome_build = 'GRCh38'")
  }
  out <- sqlquery(sql) %>%
    mutate(
      reference = na_if(reference, "NA"),
      alternate = na_if(alternate, "NA"),
      chromosome = paste0("chr", gsub("chr", "", chromosome)),
    ) %>%
    distinct() %>%
    drop_na(chromosome)
  return(out)
}

broad_acmg_encoder <- function(
  df,
  clin_sig_col = "acmg_classification",
  vus = c("variant_of_unknown_clinical_significance"),
  blb = c("benign_variant", "likely_benign_variant"),
  plp = c("pathogenic_variant", "likely_pathogenic_variant"),
  conf = c("conflicting_variant")
  ) {
  clin_sig_sym <- rlang::sym(clin_sig_col)
  df %>%
    mutate(acmg_classification_broad := case_when(
      {{clin_sig_sym}} %in% vus ~ "VUS",
      {{clin_sig_sym}} %in% blb ~ "B/LB",
      {{clin_sig_sym}} %in% plp ~ "P/LP",
      {{clin_sig_sym}} %in% conf ~ "CONFLICTING",
      TRUE ~ "Other"
    )
  ) %>%
  dplyr::select(
    chromosome,
    position,
    acmg_classification_broad
  ) %>%
  distinct() %>%
  group_by(chromosome, position) %>%
  mutate(
    acmg_classification_broad = ifelse(
      n() > 1,
      "CONFLICTING",
      acmg_classification_broad
    )
  ) %>%
  ungroup() %>%
  distinct()
}

### Flagged denovo variants (rare disease trios)
pull_denovo_platypus <- function(
  sql = NULL,
  labkey,
  chroms = c(1:22),
  sample_mappings
  ) {
  #deNovo LabKey is large.
  chroms = as.character(chroms)
  nCores <- detectCores() - 1
  outList <- mclapply(
    chroms,
    function(x) {
      sql <-  paste0("SELECT
        dn.family_id,
        dn.chrom,
        dn.position,
        dn.reference,
        dn.alternate,
        FROM denovo_flagged_variants AS dn
        WHERE dn.chrom = ('", x, "')
        AND dn.assembly = 'GRCh38'")
      sqlquery(sql)
    },
    mc.cores = nCores
  )
  names(outList) <- chroms
  out <- Reduce("rbind", outList) %>%
    mutate(chromosome = paste0("chr", chrom)) %>%
    distinct() %>%
    left_join(
      sample_mappings,
      by = "family_id" 
    ) %>%
    drop_na(sgkit_sample_id) %>%
    dplyr::select(
      chromosome,
      position,
      reference,
      alternate,
      sgkit_sample_id
    )
  return(out)
}

### Flagged denovo variants (rare disease trios)
pull_denovo_tiered <- function(
  sql = NULL,
  labkey,
  chroms = c(1:22),
  sample_mappings
  ) {
  #deNovo LabKey is large.
  chroms = as.character(chroms)
  nCores <- detectCores() - 1
  outList <- mclapply(
    chroms,
    function(x) {
      sql <-  paste0("SELECT
        tr.participant_id,
        tr.chromosome,
        tr.position,
        tr.reference,
        tr.alternate
        FROM tiering_data AS tr
        WHERE tr.chromosome = ('", x, "')
        AND tr.assembly = 'GRCh38'
        AND tr.segregation_pattern = 'deNovo'")
      sqlquery(sql)
    },
    mc.cores = nCores
  )
  names(outList) <- chroms
  out <- Reduce("rbind", outList) %>%
    mutate(chromosome = paste0("chr", chromosome)) %>%
    distinct() %>%
    left_join(
      sample_mappings,
      by = "participant_id" 
    ) %>%
    drop_na(sgkit_sample_id) %>%
    dplyr::select(
      chromosome,
      position,
      reference,
      alternate,
      sgkit_sample_id
    )
  return(out)
}

pull_sample_id_mappings <- function(
  sql = NULL,
  labkey,
  ts_sample_ids
  ) {
  if (is.null(sql)) {
    sql <- paste0("SELECT
      rda.plate_key,
      rda.participant_id,
      rda.rare_diseases_family_id
      FROM rare_disease_analysis as rda
      WHERE participant_type = 'Proband'")
  }
  out <- sqlquery(sql) %>%
    dplyr::rename(
      family_id = rare_diseases_family_id,
      sgkit_sample_id = plate_key
    ) %>%
    filter(sgkit_sample_id %in% ts_sample_ids) %>%
    distinct()
  return(out)
}

windows_overlaps <- function(
  df,
  windows_df,
  positions_only = TRUE,
  type_overlap = "any" # also within, equal etc.
) {

  ## Windows setup
  setDT(windows_df)
  setkey(windows_df, start, end)
  windows_df %<>%
    arrange(start) %>%
    mutate(yid = seq_along(start))
  
  if (positions_only) {
    ## Positions set up
    df %<>%
      arrange(position) %>%
      mutate(start = position, end = position)
  }

  setDT(df)
  df %<>%
    mutate(xid = seq_along(start))

  ## Overlaps
  overlaps <- foverlaps(
    df,
    windows_df,
    type = type_overlap,
    which = TRUE
  ) %>%
  drop_na(yid) %>%
  group_by(xid) %>%
  filter(yid == min(yid)) %>% # Some happen multiple times, take first overlap.
  distinct() %>%
  ungroup()

  df %<>%
    dplyr::rename(
      start_df = start,
      end_df = end
    ) %>%
    as_tibble()

  windows_df %<>%
    as_tibble()

  if (positions_only) {
    df %<>%
      left_join(overlaps, by = "xid") %>%
      left_join(windows_df, by = "yid") %>%
      dplyr::select(-c(xid, yid, start, end))
    return(df)

  } else {
    overlaps <- overlaps %>%
      left_join(df, by = "xid") %>%
      left_join(windows_df %>%
        dplyr::rename(
          start_window = start,
          end_window = end
        ), by = "yid") %>%
      dplyr::rename(
        start = start_df,
        end = end_df
      ) %>%
      dplyr::select(-c(xid, yid))
    return(overlaps)
  }
}


### Plots
multi_filt_cols <- function(
  df,
  filt_cols = NULL,
  filt_vals = NULL
) {
  df_filt <- df
  if (!is.null(filt_cols) & !is.null(filt_vals)) {
    ## Initial filtering
    for (filt_index in 1:length(filt_cols)) {
      filt_sym <- rlang::sym(filt_cols[filt_index])
      df_filt %<>%
        filter(
          {{filt_sym}} %in% filt_vals[[filt_index]]
        )
    }
  }
  return(df_filt)
}

## Maximum AC at the given AF threshold
max_ac_count <- function(
df,
AF_col = "AF_ts",
AC_col = "AC_ts",
AF_thresh = 0.0001
) {
  AF_col_sym <- rlang::sym(AF_col)
  AC_col_sym <- rlang::sym(AC_col)
  df %>%
   filter({{AF_col_sym}} < AF_thresh) %>%
   pull({{AC_col_sym}}) %>%
   max()
}

## Getting unrelated subset
unrelated_filter <- function(
  df,
  sample_col = "sgkit_sample_id",
  aggV2_kinship_matrix,
  kinship_coefficient = 0.0882,
  show_rel = FALSE
) {

  sample_sym <- rlang::sym(sample_col)
  ### Filter the kinship matrix to match dataframe
  kin_df <- fread(aggV2_kinship_matrix) %>%
    filter(
      IID1 %in% (df %>% pull({{sample_sym}})) &
      IID2 %in% (df %>% pull({{sample_sym}})) &
      KINSHIP > kinship_coefficient
    )

  ## Heuristic
  remove_vec <- c(kin_df$IID1, kin_df$IID2) %>% unique()
  cat(paste(
    "Number of relatives removed: ",
    length(remove_vec), "\n", sep = "")
  )
  df_norel <- df %>%
    filter(!{{sample_sym}} %in% remove_vec)

  if (show_rel == FALSE) {
    return(df_norel)
  } else if (show_rel == TRUE) {
    return(list(df_norel, remove_vec))
  }
}

gmean <- function(y) {
  y <- y[is.finite(y) & y > 0]          # geom. mean needs positive values
  if (!length(y)) return(NA_real_)
  10^(mean(log10(y), na.rm = TRUE))     # same as exp(mean(log(y)))
}

## Z SCORE DATING FUNCTION
z_normalise <- function(data, group_col, score_col, z_col, log10 = TRUE) {
  score_col <- rlang::sym(score_col)
  group_col <- rlang::sym(group_col)
  z_col <- rlang::sym(z_col)
  
  if(log10) {
    data %<>%
      mutate(
        score = log10(!!score_col)
      )
  } else {
    data %<>%
      mutate(
        score = !!score_col
      )
  }
  data %>%
    group_by(!!group_col) %>%
    mutate(!!z_col := (score - mean(score , na.rm = TRUE)) / 
                        sd(score , na.rm = TRUE)) %>%
    ungroup() %>%
    dplyr::select(-score)
}

## Wrapper for get_ind_hts + annotating with Z-scores
z_along_haps <- function(
  individual_id,
  ts_df,
  ts_list,
  ts_dir,
  ts_suffix = ".trees.tsz",
  group_col = "AC_ts",
  z_score_col = "time_z"
  ) {

  ## Individual list
  ind_list <- vector("list", length = nrow(ts_list))
  for (index in 1:nrow(ts_list)) {
    cat("Tree number: ", index, "\n")
    ind_list[[index]] <- get_ind_hts(
      paste0(ts_dir, ts_list$ts_prefix[index], ts_suffix), 
      individual_id
    )
  }
  z_score_sym <- rlang::sym(z_score_col)
  group_sym <- rlang::sym(group_col)
  ## Combine haplotypes
  ind_haps <- do.call(rbind, ind_list) |>
    filter(`1` == 1 | `0` == 1) |>
    left_join(ts_df |>
      as_tibble() |>
      dplyr::select({{group_sym}}, {{z_score_sym}}, position),
      by = "position"
    ) |>
    drop_na({{z_score_sym}}) |>
    mutate(
      mutation_hap = case_when(
        (`1` == 1 & `0` == 0) ~ "Haplotype 1",
        (`0` == 1 & `1` == 0) ~ "Haplotype 2"
      )               
    )
  ## Homozygs across both haps...
  ind_haps <- rbind(
    ind_haps |>
      drop_na(mutation_hap),
    ind_haps |>
      filter(is.na(mutation_hap)) |>
      mutate(mutation_hap = "Haplotype 1"),
    ind_haps |>
      filter(is.na(mutation_hap)) |>
      mutate(mutation_hap = "Haplotype 2")
  ) |>
  arrange(position) |>
  drop_na(c({{z_score_sym}}, mutation_hap))
}

gnn_along_haps <- function(
  individual_id,
  ts_list,
  ts_dir,
  ts_suffix = ".trees.tsz",
  pops_df,
  starts_ends_df
  ) {
  ## Individual list
  ind_list <- vector("list", length = nrow(ts_list))
  for (index in 1:nrow(ts_list)) {
    cat("Tree number: ", index, "\n")
    ind_list[[index]] <- local_gnn_sample(
      ts_path = paste0(ts_dir, ts_list$ts_prefix[index], ts_suffix),
      ind_id = individual_id,
      pops_df = pops_df
    )
  }
  ind_gnn <- do.call(rbind, ind_list) 
  ## Some re-formatting 
  gnn_out <- ind_gnn |>
    filter(left != 0 & right != max(ind_gnn$right)) |>
    arrange(haplotype, left) |>
    group_by(haplotype) |>
    mutate(
      new_right = lead(left, default = last(right)) - 1,
      right = new_right
    ) |>
    dplyr::select(-new_right) |>
    ungroup() |>
    dplyr::rename(start = left, end = right) |>
    arrange(start) |>
    windows_overlaps(
      windows_df = starts_ends_df,
      positions_only = FALSE,
      type_overlap = "any"
    ) |> 
    mutate(end = ifelse(end > end_window, end_window, end)) |>
    pivot_longer(
      !c(haplotype, start, end, ts_name, start_window, end_window),
      names_to = "pop",
      values_to = "gnn_prop"
    ) |>
    group_by(haplotype, start) |>
    mutate(
      ymax = cumsum(gnn_prop),
      ymin = ymax - gnn_prop
    ) |>
    dplyr::select(-gnn_prop) |>
    group_by(ts_name, haplotype) |>
    mutate(end = ifelse(end == max(end) & end_window > end, end_window, end)) |>
    ungroup()
}

perm_test <- function(test_set, background_set, n_permutations = 10000) {
  
  # Step 1: Define a function to calculate the observed test statistic (mean age difference)
  observed_statistic <- function(test_set, background_set) {
    exp(mean(log(test_set$mean_time))) - exp(mean(log(background_set$mean_time)))
  }
  
  # Step 2: Calculate the observed difference in mean age
  observed_diff <- observed_statistic(test_set, background_set)
  
  # Step 3: Create a combined data frame
  combined_df <- rbind(test_set, background_set)
  
  # Step 4: Initialize a vector to hold permuted statistics
  permuted_diffs <- numeric(n_permutations)
  
  # Step 5: Perform permutations
  for (i in 1:n_permutations) {
    print(i)
    permuted_df <- combined_df[sample(nrow(combined_df)), ]
    permuted_test_set <- permuted_df[1:nrow(test_set), ]
    permuted_background_set <- permuted_df[(nrow(test_set) + 1):nrow(combined_df), ]
    
    permuted_diffs[i] <- observed_statistic(permuted_test_set, permuted_background_set)
  }
  
  # Step 6: Calculate p-value based on permutation distribution
  p_value <- mean(abs(permuted_diffs) >= abs(observed_diff))
  
  return(p_value)
}

stitch_ibd_segments_by_position <- function(ibd_df, gap_threshold = 1000) {
  ibd_df %>%
    arrange(chromosome, position, ibd_left) %>%  # Ensure segments are sorted within each group
    group_by(chromosome, position) %>%
    mutate(
      # Identify groups to stitch by checking gaps and carrier consistency
      stitch_group = cumsum(
        lag(ibd_right, default = dplyr::first(ibd_right)) + gap_threshold < ibd_left |
        lag(num_carriers, default = dplyr::first(num_carriers)) != num_carriers
      )
    ) %>%
    group_by(chromosome, position, stitch_group) %>%
    summarise(
      ibd_left = min(ibd_left),
      ibd_right = max(ibd_right),
      span = ibd_right - ibd_left,
      num_carriers = dplyr::first(num_carriers),
      .groups = "drop"
    ) %>%
    ungroup()
}

# Function to interpolate cM values
interpolate_cm <- function(
  tracts,
  recomb_map,
  left = "left",
  right = "right"
) {

  left_sym <- rlang::sym(left)
  right_sym <- rlang::sym(right)
  # Ensure recomb_map is sorted by chromosome and pos
  recomb_map <- recomb_map %>% arrange(chromosome, pos)
  
  # Split recomb_map by chromosome
  recomb_map_split <- split(recomb_map, recomb_map$chromosome)
  tracts <- tracts %>% mutate(chromosome = as.character(chromosome))
  recomb_map_split <- split(
    recomb_map %>% mutate(chromosome = as.character(chromosome)), recomb_map$chromosome
  )

  tracts_nested <- tracts %>%
    arrange(chromosome, {{left_sym}}) %>%
    group_by(chromosome) %>%
    tidyr::nest()

  result <- tracts_nested %>%
    mutate(
      data = purrr::map2(
        chromosome,
        data,
        ~ {
          recomb_chr <- recomb_map_split[[as.character(.x)]]
          if (is.null(recomb_chr)) {
            stop(paste("No recombination map found for chromosome", .x))
          }
        
          .y %>%
            mutate(
              left_cm = approx(
                x = recomb_chr$pos,
                y = recomb_chr$pos_cm,
                xout = {{left_sym}},
                rule = 2
              )$y,
              right_cm = approx(
                x = recomb_chr$pos,
                y = recomb_chr$pos_cm,
                xout = {{right_sym}},
                rule = 2
              )$y,
              cM_length = right_cm - left_cm
            )
        }
      )
    ) %>%
    tidyr::unnest(data)
  
    return(result)
}

calculate_ibd_age <- function(ibd_df, from_mb = FALSE) {
  if(from_mb) {
    ibd_df$cM_length <- ibd_df$span / 1e6  # Convert bp to Mb, then use 1 Mb ≈ 1 cM
  } 

  # Calculate age in generations
  ibd_df$ibd_time <- ifelse(
    ibd_df$num_carriers > 0 & ibd_df$cM_length > 0,
    100 / (2 * ibd_df$cM_length),  # Adjust by num_carriers / 2
    NA  # Handle cases where span or carriers are zero
  )
  
  return(ibd_df)
}

outlier_logistic <- function(
  df,
  outlier_thresh = 10,
  outlier_col = "ADAC_ts",
  outcome_vec = c("0-1", "0-n1"),
  outcome_col = "gnocchi_z",
  group_vals = 2,
  group_col = "AC_ts",
  additional_terms = NULL,
  correct_groups = TRUE,
  group_as_factor = FALSE,
  direction = "greater"
) {

  group_col_sym <- rlang::sym(group_col)
  outlier_col_sym <- rlang::sym(outlier_col)
  outcome_col_sym <- rlang::sym(outcome_col)
  
  df %<>%
    filter(
        {{group_col_sym}} %in% group_vals
    ) %>%
    drop_na({{outlier_col_sym}}) %>%
    drop_na({{outlier_col_sym}})

  if (group_as_factor) {
    df %<>%
      mutate({{group_col_sym}} := as.factor({{group_col_sym}}))
  }

  if (direction == "greater") {
    df %<>%
    mutate(is_outlier = ifelse({{outlier_col_sym}} > outlier_thresh, "Yes", "No"),
           is_outlier = as.factor(is_outlier))
  } else if (direction == "lesser") {
    df %<>%
    mutate(is_outlier = ifelse({{outlier_col_sym}} < outlier_thresh, "Yes", "No"),
           is_outlier = as.factor(is_outlier))
  } else if (direction == "between") {
    df %<>%
    mutate(is_outlier = ifelse({{outlier_col_sym}} > outlier_thresh[1] & {{outlier_col_sym}} <= outlier_thresh[2],  "Yes", "No"),
           is_outlier = as.factor(is_outlier))
  }

  glm_list <- vector("list", length = length(outcome_vec))
  for (index in 1:length(outcome_vec)) {
    nsamples <- df %>%
      filter({{outcome_col_sym}} == outcome_vec[index]) %>%
      nrow()
    if (nsamples == 0) {
      cat(paste0("No samples to analyse in group: ", outcome_vec[index], "\n"))
    } else {
      df %<>%
        mutate(overlaps_outcome = ifelse({{outcome_col_sym}} == outcome_vec[index], "Yes", "No"),
               overlaps_outcome = as.factor(overlaps_outcome)
        )
      if (length(group_vals) == 1 | correct_groups == FALSE) {
        model <- df %$%
          glm(overlaps_outcome ~ is_outlier, family = "binomial")
      } else {
        glm_formula <- as.formula(
          paste0("overlaps_outcome ~ is_outlier * ", group_col)
        )
        if (!is.null(additional_terms)) {
          glm_formula <- as.formula(
            paste0("overlaps_outcome ~ is_outlier * ", group_col, " + ", additional_terms, )
          )
        }
        model <-
          glm(as.formula(glm_formula), family = "binomial", data = df)
      }
      ci_model  <- data.frame(confint.default(model))
      colnames(ci_model) <- c("conf.low", "conf.high")
      model_tidy <- model %>%
          tidy()
      upper_vec <- ci_model$conf.high[rownames(ci_model) %in% model_tidy$term]
      lower_vec <- ci_model$conf.low[rownames(ci_model) %in% model_tidy$term]

      glm_list[[index]] <- model_tidy %>%
        mutate(
          conf.high = upper_vec,
          conf.low = lower_vec,
          p.adjust = p.value * length(outcome_vec),
          p.adjust = ifelse(p.adjust > 1, 1, p.adjust),
          {{outcome_col_sym}} := outcome_vec[index]
        )
    }
  }
  glm_all <- do.call(rbind, glm_list) %>%
    mutate(
      outlier_score = paste(as.character(outlier_thresh), collapse = " - "),
      outlier_method = outlier_col,
      group := paste(group_vals, collapse = " + "),
      outcome := factor({{outcome_col_sym}},
                        levels = outcome_vec),
      direction = direction
    )
  return(glm_all)
}

outlier_logistic_loop_wrapper <- function(
  df,
  outlier_thresh_vec = c(1, 3, 10, 30),
  outlier_col,
  outcome_vec,
  outcome_col,
  group_col = "AC_ts",
  group_vals = c(1:9),
  additional_terms = NULL,
  group_as_factor = FALSE,
  correct_groups = TRUE, ## Add group_col as covariate.
  direction = "greater"
) {

  cat(paste0("Outlier column: ", outlier_col, "\n"))
  outlier_list <- list()
  for (index in 1:length(outlier_thresh_vec)) {
    if (direction == "between") {
       if (index < length(outlier_thresh_vec)) {
        outlier_thresh =  c(outlier_thresh_vec[[index]], outlier_thresh_vec[[index + 1]])
       } else {
         return(do.call(rbind, outlier_list) %>%
                  filter(term == "is_outlierYes") %>%
                  dplyr::select(-term)
          )
       }
    } else {
      outlier_thresh <- outlier_thresh_vec[[index]]
    }
    outlier_list[[index]] <- df %>%
      outlier_logistic(
        outcome_col = outcome_col,
        outcome_vec = outcome_vec,
        outlier_col = outlier_col,
        outlier_thresh = outlier_thresh,
        group_col = group_col,
        group_vals = group_vals,
        additional_terms = additional_terms,
        correct_groups = correct_groups,
        group_as_factor = group_as_factor,
        direction = direction
      )
  }
  out <- do.call(rbind, outlier_list) %>%
    filter(term == "is_outlierYes") %>%
    dplyr::select(-term)
}

outlier_logistic_twin_loop_wrapper <- function(
  df,
  outlier_thresh_vec = c(1, 3, 10, 30),
  outlier_col,
  outcome_vec,
  outcome_col,
  group_col = "AC_ts",
  group_vec = c(1:9),
  additional_terms = "pop",
  correct_groups = TRUE,
  direction = "greater"
) {
  
  cat(paste0("Outlier column: ", outlier_col, "\n"))
  outlier_list <- list()
  for (index in 1:length(outlier_thresh_vec)) {
    if (direction == "between") {
       if (index < length(outlier_thresh_vec)) {
        outlier_thresh =  c(outlier_thresh_vec[[index]], outlier_thresh_vec[[index + 1]])
       } else {
         return(do.call(rbind, outlier_list) %>%
                  filter(term == "is_outlierYes") %>%
                  dplyr::select(-term)
          )
       }
    } else {
      outlier_thresh <- outlier_thresh_vec[[index]]
    }
    group_list <- list()
    for (index_group in 1:length(group_vec)) {
      group_val <- group_vec[index_group]
      cat(paste0(
        "Outlier threshold: ", outlier_thresh,
        " Group: ", group_val, "\n"
      ))
      group_list[[index_group]] <- df %>%
        outlier_logistic(
          outcome_col = outcome_col,
          outcome_vec = outcome_vec,
          outlier_col = outlier_col,
          outlier_thresh = outlier_thresh,
          group_col = group_col,
          group_vals = group_val,
          additional_terms = additional_terms,
          correct_groups = correct_groups,
          direction = direction
        )
    }
    outlier_list[[index]] <- do.call(rbind, group_list)
  }
  out <- do.call(rbind, outlier_list) %>%
    filter(term == "is_outlierYes") %>%
    dplyr::select(-term)
}

perm_test <- function(test_set, background_set, n_permutations = 10000) {
  
  print(exp(mean(log(test_set$mean_time))))
  print(exp(mean(log(background_set$mean_time))))
  # function to calculate the observed test statistic (mean age difference)
  observed_statistic <- function(test_set, background_set) {
    exp(mean(log(test_set$mean_time))) - exp(mean(log(background_set$mean_time)))
  }
  
  observed_diff <- observed_statistic(test_set, background_set)
  combined_df <- rbind(test_set, background_set)
  permuted_diffs <- numeric(n_permutations)
  
  # Perform permutations
  for (i in 1:n_permutations) {
    permuted_df <- combined_df[sample(nrow(combined_df)), ]
    permuted_test_set <- permuted_df[1:nrow(test_set), ]
    permuted_background_set <- permuted_df[(nrow(test_set) + 1):nrow(combined_df), ]
    
    permuted_diffs[i] <- observed_statistic(permuted_test_set, permuted_background_set)
  }
  
  # Calculate p-value based on permutation distribution
  p_value <- mean(abs(permuted_diffs) >= abs(observed_diff))
  
  return(p_value)
}