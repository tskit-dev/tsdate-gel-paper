# tsdate-gel-paper
Publicly shareable code and plot data for the "Tracing the evolutionary histories of ultra-rare variants using variational dating of large ancestral recombination graphs" manuscript

> Nathaniel S. Pope, Sam Tallman, Ben Jeffery, Duncan Robertson, Yan Wong, Savita Karthikeyan, Peter L. Ralph, and Jerome Kelleher (2026) _Tracing the evolutionary histories of ultra-rare variants using variational dating of large ancestral recombination graphs_. bioRxiv: 2026.01.07.698223; doi: https://doi.org/10.64898/2026.01.07.698223

- The ``gel_analysis`` directory contains all details on the analyses performed on Genomics England
  data. Please see the README in that directory for more details.
- The ``scripts`` directory contains scripts to run benchmarks and analysis on
  data sources that are not from Genomics England. Please see the README in
  that directory for more details.
- The ``tools`` and ``makefiles`` directory contain Makefiles used by files in ``scripts``.
  Please see the README in ``scripts`` for more details.
- The ``tsinfer-snakemake`` directory contains a copy of the Snakemake workflow used to infer
  and date the ARGs. Please see the README in that directory for its source and licence.
- The ``data`` directory contains the data needed to make the plots, and ``figures`` is where
  the plots are written.

## Archive

A snapshot of this repository, including the copy of the Snakemake workflow, is archived on
Zenodo: https://doi.org/10.5281/zenodo.23142020

## Data

- Allele ages for non-rare variants (DAC > 20) in the Genomics England data: https://doi.org/10.5281/zenodo.23038476
- Inferred ARGs for chromosomes 17 and 20 of the 1000 Genomes Project: https://doi.org/10.5281/zenodo.23086785

The full allele age dataset and inferred ARGs for Genomics England are available to approved
researchers within the [Genomics England Research Environment](https://www.genomicsengland.co.uk/research).

## Software

The methods are implemented in [tsdate](https://github.com/tskit-dev/tsdate) and
[tsinfer](https://github.com/tskit-dev/tsinfer), which are available from PyPI.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`), except for
`tsinfer-snakemake`, which is distributed under its own MIT licence.

The following data files come from published work. They are included with attribution and are not
covered by the MIT licence:

- `data/Phlash_fig7b.csv`: PHLASH population size estimates from Terhorst (2025), _Nature Genetics_
  57:2570–2577, https://doi.org/10.1038/s41588-025-02323-x; included with the author's permission.
- `data/chr17inversion/chr17q21.31_time_plot.csv`: Relate-based age estimates for the 17q21.31
  inversion from Ignatieva et al. (2025), _Molecular Biology and Evolution_ 42:msaf190,
  https://doi.org/10.1093/molbev/msaf190 (also available at https://github.com/a-ignatieva/dolores-paper).
- `data/chr17inversion/donnelly_et_al_table_2.csv`: 17q21 inversion marker SNPs from Table 2 of
  Donnelly et al. (2010), _American Journal of Human Genetics_ 86:161–171.
