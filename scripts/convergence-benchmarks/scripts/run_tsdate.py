import os
import tsdate
import tskit
import tszip
import time
import numpy as np
import scipy.stats
import json

import logging
logging.basicConfig(level=logging.INFO, filename=snakemake.log.logfile)

ts = tszip.load(snakemake.input.trees)
seed = int(snakemake.wildcards.seed)
sample_size = int(snakemake.wildcards.sample_size)
ep_iterations = int(snakemake.wildcards.ep_iterations)
rescaling_iterations = int(snakemake.wildcards.rescaling_iterations)

mu = None
for provenance in reversed(ts.provenances()):
    rec = json.loads(provenance.record)
    if rec["parameters"]["command"] == "sim_mutations":
        mu = rec["parameters"]["rate"]
assert mu is not None, "No mutation rate in provenance"
logging.info(f"Using a mutation rate of {mu}")

dts, fit = tsdate.date(
    ts, 
    mutation_rate=mu,
    max_iterations=ep_iterations,
    rescaling_iterations=rescaling_iterations,
    rescaling_intervals=snakemake.params.rescaling_intervals,
    match_segregating_sites=False,
    regularise_roots=True,
    eps=snakemake.params.min_branch_length,
    set_metadata=False,
    return_fit=True,
)

node_posteriors = fit.node_posteriors()
mutation_posteriors = fit.mutation_posteriors()
np.save(snakemake.output.mutation_posteriors, mutation_posteriors)
np.save(snakemake.output.node_posteriors, node_posteriors)
np.save(snakemake.output.node_times, dts.nodes_time)
np.save(snakemake.output.mutation_times, dts.mutations_time)

