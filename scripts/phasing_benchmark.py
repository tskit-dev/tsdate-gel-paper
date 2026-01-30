"""
Show the impact of incorrectly assigning phase to singleton mutations.
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

overwrite_cache = False
true_trees = "../data/supp_benchmark_sim.tsz"
inf_trees = "../data/supp_benchmark_inf.tsz"
cache = "../data/phasing_benchmark_inf.pkl"

# --- simulate trees
trees_seed = 1024
num_samples = 20000
num_cpus = 30
if not os.path.exists(true_trees) or overwrite_cache:
    ts = msprime.sim_ancestry(
        samples=num_samples,
        sequence_length=1e8,
        recombination_rate=1e-8,
        population_size=num_samples * 2,
        model=[msprime.DiscreteTimeWrightFisher(duration=100), msprime.StandardCoalescent()],
        random_seed=trees_seed,
    )
    ts = msprime.sim_mutations(ts, rate=1.29e-8, random_seed=trees_seed+1000)
    tszip.compress(ts, true_trees)
    del ts

if not os.path.exists(inf_trees) or overwrite_cache:
    its = tsinfer.infer(tsinfer.SampleData.from_tree_sequence(ts), num_threads=num_cpus)
    tszip.compress(its, inf_trees)
    del its


# --- date with various phasing of singletons
phasing_seed = 5024
midpoint = True
ep_iter = 10
rescaling_iter = 3
rescaling_interv = 10000

def rephase_singletons(ts, method='random', use_node_times=False, random_seed=None):
    """
    Rephase singleton mutations in the tree sequence. `method='random'` assigns
    phase uniformly at random (if use_nodes_time is False) or with probability proportional
    to segment age (if use_nodes_time is True). `method='oldest_time'` assigns phase to the
    segment with the oldest parent. `method='shortest_span'` assigns phase to
    the segment with the shortest span.
    """
    assert method in ['random', 'oldest_time', 'shortest_span']
    rng = np.random.default_rng(random_seed)

    mutations_node = ts.mutations_node.copy()
    mutations_time = ts.mutations_time.copy()

    singletons = np.bitwise_and(ts.nodes_flags[mutations_node], tskit.NODE_IS_SAMPLE)
    singletons = np.flatnonzero(singletons)
    tree = ts.first()
    for i in singletons:
        position = ts.sites_position[ts.mutations_site[i]]
        individual = ts.nodes_individual[ts.mutations_node[i]]
        time = ts.nodes_time[ts.mutations_node[i]]
        assert individual != tskit.NULL
        assert time == 0.0
        tree.seek(position)
        nodes_id = ts.individual(individual).nodes
        nodes_length = []
        nodes_span = []
        for n in nodes_id:
            parent = tree.parent(n)
            assert parent != tskit.NULL
            parent_age = tree.time(parent)
            nodes_length.append(parent_age)
            edge = tree.edge(n)
            assert edge != tskit.NULL
            haplotype_span = ts.edges_right[edge] - ts.edges_left[edge]
            nodes_span.append(haplotype_span)
        if method == 'random':
            nodes_prob = nodes_length if use_node_times else np.ones(nodes_id.size)
            nodes_prob /= nodes_prob.sum()
            node = rng.choice(nodes_id, p=nodes_prob, size=1)[0]
        elif method == 'oldest_time':
            node = nodes_id[np.argmax(nodes_length)]
            assert ts.nodes_time[node] == nodes_length.max()
        elif method == 'shortest_span':
            node = nodes_id[np.argmin(nodes_span)]
        mutations_node[i] = node
        if not np.isnan(mutations_time[i]):
            parent_time = tree.time(tree.parent(mutations_node[i]))
            mutations_time[i] = (time + parent_time) / 2

    tables = ts.dump_tables()
    tables.mutations.node = mutations_node
    tables.mutations.time = mutations_time
    tables.sort()
    return tables.tree_sequence()

if not os.path.exists(cache) or overwrite_cache:
    import tsdate
    print(tsdate.__version__)
    ts = tszip.decompress(inf_trees)
    ts0 = tszip.decompress(true_trees)

    # for inferred only
    ts = tsdate.preprocess_ts(ts)
    multimapped = np.bincount(ts.mutations_site, minlength=ts.num_sites) > 1
    ts = ts.delete_sites(np.flatnonzero(multimapped))

    # true midpoint ages of singletons
    parents = np.full(ts.num_mutations, -1)
    freq = np.full(ts.num_mutations, -1)
    for t in ts.trees():
        for m in t.mutations():
            if m.edge != tskit.NULL:
                parents[m.id] = t.parent(m.node)
                freq[m.id] = t.num_samples(m.node)
    singletons = freq == 1
    parents = parents[singletons]

    true_ages = {}
    true_midpoint_ages = {}
    for t in ts0.trees():
        for s in t.sites():
            if len(s.mutations) == 1:
                true_ages[int(s.position)] = s.mutations[0].time
                true_midpoint_ages[int(s.position)] = (
                    ts0.nodes_time[ts0.edges_parent[s.mutations[0].edge]] +
                    ts0.nodes_time[ts0.edges_child[s.mutations[0].edge]] 
                ) / 2
    true = np.full(ts.num_mutations, np.nan)
    true_mid = np.full(ts.num_mutations, np.nan)
    for m in ts.mutations():
        pos = int(ts.sites_position[m.site])
        if pos in true_ages:
            true[m.id] = true_ages[pos]
            true_mid[m.id] = true_midpoint_ages[pos]
    true = true[singletons]
    true_mid = true_mid[singletons]

    # different datings
    ts_phase, fit_phase = tsdate.date(
        ts, 
        mutation_rate=1.29e-8, 
        singletons_phased=True, 
        max_iterations=ep_iter,
        rescaling_iterations=rescaling_iter,
        rescaling_intervals=rescaling_interv,
        set_metadata=False,
        return_fit=True,
    )

    ts_random_phase = rephase_singletons(
        ts, method='random', use_node_times=False, random_seed=phasing_seed+2,
    )
    print(
        "bad phase (random assignment):", 
        sum(ts_random_phase.mutations_node[singletons] != ts.mutations_node[singletons])/sum(singletons),
    )
    ts_random_phase, fit_random_phase = tsdate.date(
        ts_random_phase,
        mutation_rate=1.29e-8,
        singletons_phased=True,
        max_iterations=ep_iter,
        rescaling_iterations=rescaling_iter,
        rescaling_intervals=rescaling_interv,
        set_metadata=False,
        return_fit=True,
    )

    ts_shortest_span = rephase_singletons(
        ts, method='shortest_span', random_seed=phasing_seed+3,
    )
    print(
        "bad phase (shortest span):", 
        sum(ts_shortest_span.mutations_node[singletons] != ts.mutations_node[singletons])/sum(singletons),
    )
    ts_shortest_span, fit_shortest_span = tsdate.date(
        ts_shortest_span,
        mutation_rate=1.29e-8,
        singletons_phased=True,
        max_iterations=ep_iter,
        rescaling_iterations=rescaling_iter,
        rescaling_intervals=rescaling_interv,
        set_metadata=False,
        return_fit=True,
    )

    ts_no_phase, fit_no_phase = tsdate.date(
        ts, mutation_rate=1.29e-8, 
        singletons_phased=False,
        max_iterations=ep_iter,
        rescaling_iterations=rescaling_iter,
        rescaling_intervals=rescaling_interv,
        set_metadata=False,
        return_fit=True,
    )

    point_estimates = {
        "true": true,
        "true_midpoint": true_mid,
        "phased": ts_phase.mutations_time[singletons],
        "random_phase": ts_random_phase.mutations_time[singletons],
        "shortest_span": ts_shortest_span.mutations_time[singletons],
        "no_phase": ts_no_phase.mutations_time[singletons],
    }
    ages = {
        "point_estimates": point_estimates,
        "true": true,
        "true_midpoint": true_mid,
        "phased": fit_phase.mutation_posteriors()["mean"][singletons],
        "random_phase": fit_random_phase.mutation_posteriors()["mean"][singletons],
        "shortest_span": fit_shortest_span.mutation_posteriors()["mean"][singletons],
        "no_phase": fit_no_phase.mutation_posteriors()["mean"][singletons],
    }
    pickle.dump(ages, open(cache, "wb"))
else:
    ages = pickle.load(open(cache, "rb"))


# --- make figure
if midpoint:
    true = ages["true_midpoint"]
else:
    true = ages["true"]
phased = ages["phased"]
random_phase = ages["random_phase"]
shortest_span = ages["shortest_span"]
no_phase = ages["no_phase"]

plot_path = "../figures/phasing_benchmark.pdf"
rows = 1
cols = 4
fig, axs = plt.subplots(
    rows, cols,
    figsize=(cols * 2.5, rows * 2.75), 
    constrained_layout=True, sharey=True, sharex=True,
)

def stats(x, y):
    ok = np.logical_and(np.isfinite(x), np.isfinite(y))
    r = np.corrcoef(x[ok], y[ok])[0,1]
    rmse = np.sqrt(np.mean((x[ok] - y[ok])**2))
    return r, rmse

xm = np.nanmean(true)
xmp = np.nanmean(true) + 1
r, rmse = stats(np.log10(true), np.log10(phased))
axs[0].text(0.01, 0.99, f"Known phase\n$r={r:.3f}$", size=10, transform=axs[0].transAxes, ha='left', va='top')
axs[0].hexbin(true, phased, mincnt=1, xscale="log", yscale="log")
axs[0].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
axs[0].set_title("A.", loc="left", fontweight="bold")
r, rmse = stats(np.log10(true), np.log10(random_phase))
axs[1].text(0.01, 0.99, f"Random phase\n$r={r:.3f}$", size=10, transform=axs[1].transAxes, ha='left', va='top')
axs[1].hexbin(true, random_phase, mincnt=1, xscale="log", yscale="log")
axs[1].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
axs[1].set_title("B.", loc="left", fontweight="bold")
r, rmse = stats(np.log10(true), np.log10(shortest_span))
axs[2].text(0.01, 0.99, f"Shortest span\n$r={r:.3f}$", size=10, transform=axs[2].transAxes, ha='left', va='top')
axs[2].hexbin(true, shortest_span, mincnt=1, xscale="log", yscale="log")
axs[2].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
axs[2].set_title("C.", loc="left", fontweight="bold")
r, rmse = stats(np.log10(true), np.log10(no_phase))
axs[3].text(0.01, 0.99, f"Phase agnostic\n$r={r:.3f}$", size=10, transform=axs[3].transAxes, ha='left', va='top')
axs[3].hexbin(true, no_phase, mincnt=1, xscale="log", yscale="log")
axs[3].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
axs[3].set_title("D.", loc="left", fontweight="bold")

fig.supylabel("Estimated age", size=10)
if midpoint:
    fig.supxlabel("True midpoint age (unphased singleton mutations)", size=10)
else:
    fig.supxlabel("True age (unphased singleton mutations)", size=10)

plt.savefig(plot_path)


            





