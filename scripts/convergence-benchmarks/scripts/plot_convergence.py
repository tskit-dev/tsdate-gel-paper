import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rcParams["figure.dpi"] = snakemake.params.dpi
metrics = pd.read_csv(snakemake.input.csv)

fig, axs = plt.subplots(
    1, 1, 
    figsize=snakemake.params.figsize, 
    constrained_layout=True,
    sharex=True,
)

num_samples = np.unique(metrics["num_samples"].to_numpy())
seeds = np.unique(metrics["seed"].to_numpy())
iterations = np.unique(metrics["ep_iterations"].to_numpy())
cmap = plt.get_cmap("viridis")
for i, samples in enumerate(num_samples):
    relative_rmse = np.zeros(iterations.size)
    for seed in seeds:
        subset = np.logical_and(metrics["num_samples"] == samples, metrics["seed"] == seed)
        rmse = metrics.loc[subset, "node_rmse"].to_numpy()
        relative_rmse += rmse - rmse.min()
    relative_rmse /= len(seeds)
    plt.plot(iterations, relative_rmse, "-", color=cmap(i/(num_samples.size-1)), label=f"{samples}")
axs.set_ylabel("RMSE $-$ final RMSE")
axs.set_xlabel("EP iteration")
axs.legend(title="# samples", frameon=False)
plt.savefig(snakemake.output.plot)
