import tszip
import tskit
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, filename=snakemake.log.logfile, filemode='a')

seed = int(snakemake.wildcards.seed)
sample_size = int(snakemake.wildcards.sample_size)
ep_iterations = int(snakemake.wildcards.ep_iterations)
rescaling_iterations = int(snakemake.wildcards.rescaling_iterations)

ts = tszip.load(snakemake.input.trees)
node_times = np.load(snakemake.input.node_times)
mutation_times = np.load(snakemake.input.mutation_times)

node_is_sample = np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE).astype(bool)
mutation_segregating = np.array([m.edge != tskit.NULL for m in ts.mutations()])

infr_node_times = np.log10(node_times[~node_is_sample])
true_node_times = np.log10(ts.nodes_time[~node_is_sample])
node_error = infr_node_times - true_node_times
node_rmse = np.sqrt(np.mean(node_error ** 2))
node_mae = np.mean(np.abs(node_error))
node_bias = np.mean(node_error)
node_r2 = np.corrcoef(infr_node_times, true_node_times)[0, 1] ** 2

infr_mutation_times = np.log10(mutation_times[mutation_segregating])
true_mutation_times = np.log10(ts.mutations_time[mutation_segregating])
mutation_error = infr_mutation_times - true_mutation_times
mutation_rmse = np.sqrt(np.mean(mutation_error ** 2))
mutation_mae = np.mean(np.abs(mutation_error))
mutation_bias = np.mean(mutation_error)
mutation_r2 = np.corrcoef(infr_mutation_times, true_mutation_times)[0, 1] ** 2

metrics = {
    "node_rmse": node_rmse, "node_mae": node_mae, "node_bias": node_bias, "node_r2": node_r2, 
    "mutation_rmse": mutation_rmse, "mutation_mae": mutation_mae, "mutation_bias": mutation_bias, "mutation_r2": mutation_r2, 
    "num_nodes": ts.num_nodes, "num_edges": ts.num_edges, 
    "num_samples": ts.num_samples, "num_mutations": ts.num_mutations,
    "seed": seed, "ep_iterations": ep_iterations, "rescaling_iterations": rescaling_iterations,
}
logging.info(f"{metrics}")
pickle.dump(metrics, open(snakemake.output.metrics, "wb"))

