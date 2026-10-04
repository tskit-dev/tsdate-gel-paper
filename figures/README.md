Output directory for figures. The figures themselves are not included in this repository; see the paper.

Most figures that use simulated or public data can be produced by running `scripts/plot.py` and specifying the figure name (e.g. `python scripts/plot.py sampling_sim`, which produces `figures/sampling_sim+60000.pdf`). If the data files needed for a figure do not already exist in `data/`, they can be created by running `scripts/make_figure_data.py` with the same figure name; this may require downloading files or running other scripts first. See `scripts/README.md` for the full list, including the Snakemake pipelines and standalone scripts, which write their outputs elsewhere.

The R Markdown files in `gel_analysis/` also write two figures here (`am_pa-ecdf-by-dac.pdf` and `am_pa-auroc-by-dac.pdf`), but they can only be run within the Genomics England Research Environment.
