"""
Aims to show importance of long recent haplotypes in dating recent mutations
- Simulate large sequence
- Pick focal subset of singleton mutations in center of contig
- Date with progressively larger flanks around focal subset
- This is done with phase unknown, as this is the case we use in practice
"""

import os
import numpy as np
import tskit
import msprime
import pickle
import tszip
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)


# --- simulate data
seed = 1024
num_samples = 20000
length_grid = [1, 5, 10, 20, 40, 60, 100]
overwrite_cache = False

cache = "../data/sequence_length_accuracy_benchmark.tsz"
if not os.path.exists(cache) or overwrite_cache:
    ts = msprime.sim_ancestry(
        samples=num_samples,
        sequence_length=length_grid[-1] * 1e6,
        recombination_rate=1e-8,
        population_size=num_samples * 2,
        model=[msprime.DiscreteTimeWrightFisher(duration=100), msprime.StandardCoalescent()],
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=1.29e-8, random_seed=seed+1000)
    tszip.compress(ts, cache)
else:
    ts = tszip.decompress(cache)


# NB: we're not using tsinfer here, to demonstrate the issue is
# strictly a dating issue. The rationale carries over to inferred trees,
# but there will be more noise (of course).


# extract focal mutations in central window (singletons only)
intr = [
    max(0, ts.sequence_length/2 - length_grid[0] * 1e6 / 2), 
    min(ts.sequence_length, ts.sequence_length/2 + length_grid[0] * 1e6 / 2)
]
ts_focal = ts.keep_intervals([intr])
biallelic = np.bincount(ts_focal.mutations_site, minlength=ts_focal.num_sites) == 1
singleton = np.full(ts_focal.num_sites, False)
singleton[ts_focal.mutations_site[ts_focal.mutations_node < ts_focal.num_samples]] = True  # collisions don't matter, multiallelic will be filtered out
positions = ts_focal.sites_position[np.logical_and(singleton, biallelic)]
position_map = {p:i for i,p in enumerate(positions)}

true_ages = np.full(positions.size, np.nan)  # midpoint ages
for m in ts_focal.mutations():
    pos = ts_focal.sites_position[m.site]
    if pos in position_map:
        true_ages[position_map[pos]] = ts_focal.nodes_time[ts_focal.edges_parent[m.edge]] / 2

true_span = np.full(positions.size, np.nan)  # edge span
for m in ts.mutations():  
    pos = ts.sites_position[m.site]
    if pos in position_map:
        true_span[position_map[pos]] = ts.edges_right[m.edge] - ts.edges_left[m.edge]


# --- date across various truncations of the sequence, extract estimates for focal mutations
cache = "../data/sequence_length_accuracy_benchmark.pkl"
if not os.path.exists(cache) or overwrite_cache:
    import tsdate
    midpoint = ts.sequence_length / 2
    focal_mut_ages = []
    for length in length_grid:
        intr = [
            max(0, midpoint - length * 1e6 / 2), 
            min(ts.sequence_length, midpoint + length * 1e6 / 2)
        ]
        ts_trunc = ts.keep_intervals([intr])
        ts_trunc = tsdate.date(ts_trunc, mutation_rate=1.29e-8, singletons_phased=False)
        mut_ages = np.full(positions.size, np.nan)
        for m in ts_trunc.mutations():
            pos = ts_trunc.sites_position[m.site]
            if pos in position_map:
                mut_ages[position_map[pos]] = m.time
        focal_mut_ages.append(mut_ages)
    focal_mut_ages = np.stack(focal_mut_ages)
    pickle.dump({
        "focal_mut_ages": focal_mut_ages,
        "true_ages": true_ages,
        "length_grid": length_grid,
    }, open(cache, "wb"))
else:
    tmp = pickle.load(open(cache, "rb"))
    focal_mut_ages = tmp["focal_mut_ages"]
    true_ages = tmp["true_ages"]
    length_grid = tmp["length_grid"]


# --- make figures
plot_path = "../figures/sequence_length_accuracy_benchmark.pdf"
rows = 2
cols = 3
fig = plt.figure(figsize=(cols * 3, rows * 3), constrained_layout=True)
fig_top, fig_bot = fig.subfigures(2, 1, hspace=0.05, height_ratios=[1, 1])
axs = []
axs.append(fig_top.add_subplot(1, 3, 1))
axs.append(fig_top.add_subplot(1, 3, 2, sharex=axs[0], sharey=axs[0]))
axs.append(fig_top.add_subplot(1, 3, 3, sharex=axs[0], sharey=axs[0]))
bot = []
bot.append(fig_bot.add_subplot(1, 3, 1))
bot.append(fig_bot.add_subplot(1, 3, 2, sharex=bot[0], sharey=bot[0]))
bot.append(fig_bot.add_subplot(1, 3, 3, sharex=bot[0], sharey=bot[0]))

# true vs estimated age
xm = true_ages.mean()
xmp = true_ages.mean() + 1
axs[0].set_title(f"No flanking sequence", size=10)
axs[0].hexbin(true_ages, focal_mut_ages[0], mincnt=1, xscale="log", yscale="log")
axs[0].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
axs[1].set_title(f"19Mb flanking sequence", size=10)
axs[1].hexbin(true_ages, focal_mut_ages[3], mincnt=1, xscale="log", yscale="log")
axs[1].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
axs[1].get_yaxis().set_visible(False)
axs[2].set_title(f"99Mb flanking sequence", size=10)
axs[2].hexbin(true_ages, focal_mut_ages[6], mincnt=1, xscale="log", yscale="log")
axs[2].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
axs[2].get_yaxis().set_visible(False)
fig_top.suptitle("A", size=14, fontweight="bold", x=0.02, y=1.00)
fig_top.supylabel("Posterior mean age", size=10)
fig_top.supxlabel("True age (unphased singletons)", size=10)

# dating error vs span
errors = np.log10(focal_mut_ages) - np.log10(true_ages[np.newaxis, :])
bot[0].set_title(f"No flanking sequence", size=10)
bot[0].axvspan(1e6, 3e8, alpha=0.15, color="gray")
bot[0].hexbin(true_span, errors[0], mincnt=1, xscale="log")
bot[0].axhline(0, linestyle="dashed", color="red")
bot[0].text(1.1e6, 2.15, f"truncated to 1Mb\nfor dating", ha="left", va="top", color="gray")
bot[0].set_xlim(1.5e3, 3e8)
bot[0].set_ylim(-0.5, 2.2)
bot[1].set_title(f"19Mb flanking sequence", size=10)
bot[1].axvspan(20e6, 3e8, alpha=0.15, color="gray")
bot[1].hexbin(true_span, errors[3], mincnt=1, xscale="log")
bot[1].axhline(0, linestyle="dashed", color="red")
bot[1].text(2.1e7, 2.15, f"to 20Mb", ha="left", va="top", color="gray")
bot[1].get_yaxis().set_visible(False)
bot[2].set_title(f"99Mb flanking sequence", size=10)
bot[2].hexbin(true_span, errors[6], mincnt=1, xscale="log")
bot[2].axhline(0, linestyle="dashed", color="red")
bot[2].get_yaxis().set_visible(False)
fig_bot.suptitle("B", size=14, fontweight="bold", x=0.02, y=1.00)
fig_bot.supylabel(r"log$_{10}$ relative error in age", size=10)
fig_bot.supxlabel("Span in full sequence (unphased singletons)", size=10)

plt.savefig(plot_path)

