"""
Adapted from the `polegon_master` wrapper (commit 0c088b339 in
yundeng98/polegon); import I/O functions from that script and recreate the
subprocess invocation, so as to profile memory use in the binary
"""

import os
import tskit
import tszip
import time
import json
import subprocess
import numpy as np

import memray
from memray._memray import compute_statistics

import logging
logging.basicConfig(level=logging.INFO, filename=snakemake.log.logfile)


# read in functions from `polegon_master`; we do this rather than run the
# wrapper directly so as to be able to profile memory usage of the binary
import types
import importlib
loader = importlib.machinery.SourceFileLoader("polegon_master", snakemake.input.polegon_master)
polegon_master = types.ModuleType(loader.name)
loader.exec_module(polegon_master)


ts = tszip.load(snakemake.input.trees)
seed = int(snakemake.wildcards.seed)
sample_size = int(snakemake.wildcards.sample_size)
profile = snakemake.wildcards.profile
algorithm = snakemake.wildcards.algorithm

mu = None
for provenance in reversed(ts.provenances()):
    rec = json.loads(provenance.record)
    if rec["parameters"]["command"] == "sim_mutations":
        mu = rec["parameters"]["rate"]
assert mu is not None, "No mutation rate in provenance"
logging.info(f"Using a mutation rate of {mu}")

num_samples = snakemake.params.num_samples
burn_in = snakemake.params.burn_in
scaling_rep = snakemake.params.scaling_rep
thin = snakemake.params.thin
max_step = snakemake.params.max_step

input_prefix = snakemake.output.metrics.removesuffix(".pkl") + ".polegon"
output_prefix = input_prefix  # not used afaict

if profile == "peak_memory":
    memray_profile = snakemake.output.metrics.removesuffix(".pkl") + ".bin"
    if os.path.exists(memray_profile): os.remove(memray_profile)
    with memray.Tracker(memray_profile):
        Ne = ts.diversity() / (2 * mu)
        polegon_master.write_arg(ts, f"{input_prefix}_nodes.txt", f"{input_prefix}_branches.txt")
        polegon_master.write_muts(ts, f"{input_prefix}_muts.txt")
        polegon_command = [
            "/usr/bin/time", "-f %M",
            snakemake.input.polegon, 
            "-m", f"{mu}", 
            "-Ne", f"{Ne}", 
            "-burn_in", f"{burn_in}", 
            "-num_samples", f"{num_samples}", 
            "-thin", f"{thin}", 
            "-scaling_rep", f"{scaling_rep}", 
            "-input", f"{input_prefix}", 
            "-output", f"{output_prefix}", 
            "-max_step", f"{max_step}",
        ]
        result = subprocess.run(polegon_command, capture_output=True, text=True)
        assert result.returncode == 0, "Error running POLEGON"
        dts = polegon_master.read_mts(
            f"{input_prefix}_new_nodes.txt", 
            f"{input_prefix}_branches.txt", 
            f"{input_prefix}_muts.txt",
        )
    memray_stats = compute_statistics(memray_profile)
    os.remove(memray_profile)
    peak_memory_mb_pyt = memray_stats.peak_memory_allocated / 1e6
    peak_memory_mb_cpp = int(result.stderr) / 1e3
    logging.info(f"Mem usage: python {peak_memory_mb_pyt}, cpp {peak_memory_mb_cpp}")
    peak_memory_mb = max(peak_memory_mb_pyt, peak_memory_mb_cpp)
    timing_sec = np.nan
elif profile == "walltime":
    start_time = time.time()
    Ne = ts.diversity() / (2 * mu)
    polegon_master.write_arg(ts, f"{input_prefix}_nodes.txt", f"{input_prefix}_branches.txt")
    polegon_master.write_muts(ts, f"{input_prefix}_muts.txt")
    polegon_command = [
        snakemake.input.polegon, 
        "-m", f"{mu}", 
        "-Ne", f"{Ne}", 
        "-burn_in", f"{burn_in}", 
        "-num_samples", f"{num_samples}", 
        "-thin", f"{thin}", 
        "-scaling_rep", f"{scaling_rep}", 
        "-input", f"{input_prefix}", 
        "-output", f"{output_prefix}", 
        "-max_step", f"{max_step}",
    ]
    result = subprocess.run(polegon_command, capture_output=True, text=True)
    assert result.returncode == 0, "Error running POLEGON"
    dts = polegon_master.read_mts(
        f"{input_prefix}_new_nodes.txt", 
        f"{input_prefix}_branches.txt", 
        f"{input_prefix}_muts.txt",
    )
    peak_memory_mb = np.nan
    timing_sec = time.time() - start_time
else:
    raise ValueError(f"No match to {profile}")

logging.info(f"POLEGON log:\n{result.stdout}\n")

for suffix in ["new_nodes.txt", "nodes.txt", "branches.txt", "muts.txt"]:
    if os.path.exists(f"{input_prefix}_{suffix}"): 
        os.remove(f"{input_prefix}_{suffix}")

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
