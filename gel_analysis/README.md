# Analysis performed on the Genomics England RE environment

This directory contains code for analysing mutation ages inferred for participants in the Genomics England Research Environment.
You can gain access to the research environment by following these steps: [GEL Research Environment](https://www.genomicsengland.co.uk/research).

The code runs only inside the Research Environment: the paths in `config/config.yaml` point to locations there. Shared helper functions are in `functions/`.

## HTML reports
Rendered HTML reports for each section are included in this directory (download and open them in a browser):

- [GEL, Summary statistics](gel-summary_stats.html) (`gel-summary_stats.Rmd`; Figs. S11–S17)
- [GEL, Allele ages and ancestral diversity](gel-ages_ancestry.html) (`gel-ages_ancestry.Rmd`; Figs. S18–S20)
- [GEL, Allele ages and negative selection](gel-ages_negselection.html) (`gel-ages_negselection.Rmd`; ED Figs. 8–9, Fig. S21)
- [GEL, Allele ages of clinically classified mutations](gel-ages_clinical.html) (`gel-ages_clinical.Rmd`; Figs. S22–S23)

There are also reports for generating annotated parquet dataframes from inferred ARGs (per chromosome), produced by `gel-ts_df.Rmd`.
These are stored in the `chrom_reports/` subdirectory.

The `data/` subdirectory contains summary outputs that have been exported from the Research Environment: the logistic models of age by score bin (`*_age_z_logistic.csv`, written by `gel-ages_negselection.Rmd`; Tables S9–S12), and the Relate age summaries (`relate_means.csv`, written by `gel-summary_stats.Rmd`).

## Data access

For access to the complete file set required replicate these analyses within the Research Environment, please email samuel.tallman@genomicsengland.co.uk
