import os
import tsdate
import tskit
import tszip
import time
import numpy as np
import json

import memray
from memray._memray import compute_statistics

import logging
logging.basicConfig(level=logging.INFO, filename=snakemake.log.logfile)

# inside-outside uses a lot of memory, and snakemake does not enforce memory limits, 
# so set a hard limit here to avoid locking up
import resource
if hasattr(snakemake.resources, "mem_mb"):
    max_memory_bytes = snakemake.resources.mem_mb * int(1e6)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))


ts = tszip.load(snakemake.input.trees)
seed = int(snakemake.wildcards.seed)
sample_size = int(snakemake.wildcards.sample_size)
algorithm = snakemake.wildcards.algorithm
profile = snakemake.wildcards.profile

mu = None
for provenance in reversed(ts.provenances()):
    rec = json.loads(provenance.record)
    if rec["parameters"]["command"] == "sim_mutations":
        mu = rec["parameters"]["rate"]
assert mu is not None, "No mutation rate in provenance"
logging.info(f"Using a mutation rate of {mu}")


def run_benchmark():
    if algorithm == "variational_gamma":
        dts = tsdate.date(
            ts, 
            mutation_rate=mu,
            max_iterations=snakemake.params.ep_iterations, 
            rescaling_intervals=snakemake.params.rescaling_intervals,
            rescaling_iterations=snakemake.params.rescaling_iterations,
            set_metadata=False,
        )
    elif algorithm == "inside_outside":
        ne_estimate = ts.diversity() / (4 * mu)
        dts = tsdate.date(
            ts, 
            mutation_rate=mu,
            population_size=ne_estimate,
            method="inside_outside",
            set_metadata=False,
        )
    else:
        raise ValueError(f"No match to {algorithm}")
    return dts

if profile == "peak_memory":
    memray_profile = snakemake.output.metrics.removesuffix(".pkl") + ".bin"
    if os.path.exists(memray_profile): os.remove(memray_profile)
    with memray.Tracker(memray_profile):
        dts = run_benchmark()
    memray_stats = compute_statistics(memray_profile)
    os.remove(memray_profile)
    peak_memory_mb = memray_stats.peak_memory_allocated / 1e6
    timing_sec = np.nan
elif profile == "walltime":
    start_time = time.time()
    dts = run_benchmark()
    timing_sec = float(time.time() - start_time)
    peak_memory_mb = np.nan
else:
    raise ValueError(f"No match to {profile}")

true_nodes_time = np.load(snakemake.input.true_nodes_time)
assert true_nodes_time.size == dts.num_nodes
infr_nodes_time = np.log10(dts.nodes_time[true_nodes_time > 0])
true_nodes_time = np.log10(true_nodes_time[true_nodes_time > 0])
error = infr_nodes_time - true_nodes_time
rmse = np.sqrt(np.mean(error ** 2))
mae = np.mean(np.abs(error))
bias = np.mean(error)
r2 = np.corrcoef(infr_nodes_time, true_nodes_time)[0, 1] ** 2

metrics = {
    "rmse": rmse, "mae": mae, "bias": bias, "r2": r2, 
    "walltime_sec": timing_sec, "peak_memory_mb": peak_memory_mb,
    "num_nodes": ts.num_nodes, "num_edges": ts.num_edges, 
    "num_samples": ts.num_samples, "num_mutations": ts.num_mutations,
    "seed": seed, "algorithm": algorithm, "profile": profile,
}
logging.info(f"{metrics}")
pickle.dump(metrics, open(snakemake.output.metrics, "wb"))
np.save(snakemake.output.infr_nodes_time, dts.nodes_time)
