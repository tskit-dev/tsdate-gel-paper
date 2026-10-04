# tsinfer-snakemake

A Snakemake workflow for distributed ARG inference with tsinfer and dating with tsdate, used
to infer the Genomics England and 1000 Genomes Project ARGs in the paper.

This is a copy of the code files from https://github.com/benjeffery/tsinfer-snakemake at commit
[`e19873b`](https://github.com/benjeffery/tsinfer-snakemake/tree/e19873b) (2025-08-11),
included here so that it is archived with the rest of the analysis code. The workflow is
distributed under the MIT licence in `LICENSE` (copyright Tskit Developers).

The upstream `test_data/` (used by `test_config.yaml`) and `useful_data/` (e.g. the HapMapII
GRCh38 genetic maps) directories are not included; they can be obtained from the upstream
repository at the same commit.

To configure a run, copy `config.yaml.example` to `config.yaml` and edit it; see the upstream
repository for further details.
