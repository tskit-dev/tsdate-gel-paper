import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams["figure.dpi"] = snakemake.params.dpi
metrics = pd.read_csv(snakemake.input.csv)

fig, axs = plt.subplots(
    1, 3, 
    figsize=snakemake.params.figsize, 
    constrained_layout=True,
    sharex=True,
)

sns.lineplot(
    data=metrics[metrics["profile"] == "walltime"],
    x="num_samples",
    y="walltime_sec",
    hue="algorithm",
    estimator="median",
    errorbar=None,
    legend=True,
    ax=axs[0],
)
axs[0].axhline(y=86400, linestyle="dashed", color="black", linewidth=0.5)
axs[0].text(10, 86400, "24 hours", va="bottom", ha="left", color="black")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].tick_params(which="both")
axs[0].set_ylabel("Wall time (seconds)")
axs[0].set_xlabel("")

sns.lineplot(
    data=metrics[metrics["profile"] == "peak_memory"],
    x="num_samples",
    y="peak_memory_mb",
    hue="algorithm",
    estimator="median",
    errorbar=None,
    legend=False,
    ax=axs[1],
)
axs[1].axhline(y=1e5, linestyle="dashed", color="black", linewidth=0.5)
axs[1].text(10, 1e5, "100 Gb", va="bottom", ha="left", color="black")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].minorticks_on()
axs[1].set_ylabel("Peak memory (Mb)")
axs[1].set_xlabel("")

sns.lineplot(
    data=metrics[metrics["profile"] == "walltime"],
    x="num_samples",
    y="rmse",
    hue="algorithm",
    estimator="median",
    errorbar=None,
    legend=False,
    ax=axs[2],
)
axs[2].set_xscale("log")
axs[2].set_ylabel("Error (RMSE)")
axs[2].set_xlabel("")

fig.supxlabel("Number of samples")
handles, labels = axs[0].get_legend_handles_labels() 
labels = [x.replace("_", " ") for x in labels]
axs[0].get_legend().remove()
fig.legend(
    handles, labels,
    loc="outside upper center",
    frameon=False,
    ncol=3,
)
plt.savefig(snakemake.output.plot)
