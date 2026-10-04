"""
Show the (indirect) impact of singleton phasing on deeper parts of the
genealogy, using doubleton mutation ages as the readout. Singletons are
rephased under several conditions (as in phasing_benchmark_uncert.py), but
doubletons are never rephased directly -- so their age estimates probe how far
singleton mis-phasing propagates into the nearest internal structure. Mutation
ages are used (rather than node ages) because mutations are position-indexed
and so port directly between the true and inferred tree sequences.
"""

import os
import numpy as np
import tskit
import msprime
import pickle
import tszip
import scipy.stats
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

overwrite_cache = False
true_trees = "../data/supp_benchmark_sim.tsz"
inf_trees = "../data/supp_benchmark_inf.tsz"
cache = "../data/phasing_benchmark_doubleton.pkl"

# nominal posterior interval widths at which to evaluate coverage
interval_width = np.array(
    [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

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


# the singleton-phasing conditions shown in the figure (key, panel label, panel letter)
conditions = [
    ("known", "True singleton phase", "A."),
    ("random", "Random singleton phase", "B."),
    ("shortest_span", "Shortest span", "C."),
    ("no_phase", "Phase agnostic", "D."),
]

def date_phasing(ts, condition, random_seed):
    """
    Date `ts` under one of the singleton-phasing conditions and return
    `(dated_ts, fit)`. `'known'` keeps the existing (correct) phase, `'random'`
    and `'shortest_span'` rephase singletons accordingly, and `'no_phase'`
    dates phase-agnostically.
    """
    singletons_phased = True
    if condition == "known":
        d = ts
    elif condition == "random":
        d = rephase_singletons(ts, method="random", use_node_times=False, random_seed=random_seed)
    elif condition == "shortest_span":
        d = rephase_singletons(ts, method="shortest_span", random_seed=random_seed)
    elif condition == "no_phase":
        d = ts
        singletons_phased = False
    else:
        raise ValueError(condition)
    return tsdate.date(
        d,
        mutation_rate=1.29e-8,
        singletons_phased=singletons_phased,
        max_iterations=ep_iter,
        rescaling_iterations=rescaling_iter,
        rescaling_intervals=rescaling_interv,
        set_metadata=False,
        return_fit=True,
    )

def doubleton_stats_by_position(ts, fit, positions):
    """
    For each integer position in `positions`, extract from the biallelic sites of
    the dated `ts`/`fit`: the mutation-age posterior (`mean`, `variance`) and the
    posteriors of the two nodes bounding the mutation's branch -- the doubleton
    node below the mutation (`child_*`) and its parent (`parent_*`). The two node
    posteriors allow the exact within-branch age to be reconstructed by sampling
    (the mutation age is uniform between the two node times). Entries with no
    matching (biallelic) site are `nan`.
    """
    mpost = fit.mutation_posteriors()
    npost = fit.node_posteriors()
    muts_per_site = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    keys = ["mean", "variance", "child_mean", "child_variance", "parent_mean", "parent_variance"]
    record = {}
    for m in ts.mutations():
        if muts_per_site[m.site] != 1 or m.edge == tskit.NULL:
            continue
        pos = int(ts.sites_position[m.site])
        child = m.node
        parent = ts.edges_parent[m.edge]
        record[pos] = (
            mpost["mean"][m.id], mpost["variance"][m.id],
            npost["mean"][child], npost["variance"][child],
            npost["mean"][parent], npost["variance"][parent],
        )
    out = {k: np.full(len(positions), np.nan) for k in keys}
    for i, p in enumerate(positions):
        if p in record:
            for k, val in zip(keys, record[p]):
                out[k][i] = val
    return out


if not os.path.exists(cache) or overwrite_cache:
    import tsdate
    print(tsdate.__version__)
    ts = tszip.decompress(inf_trees)
    ts0 = tszip.decompress(true_trees)

    # for inferred only
    ts = tsdate.preprocess_ts(ts)
    multimapped = np.bincount(ts.mutations_site, minlength=ts.num_sites) > 1
    ts = ts.delete_sites(np.flatnonzero(multimapped))

    # doubletons (defined on the inferred trees) and their genomic positions; this set
    # is identical across phasing conditions, since rephasing only touches singletons
    freq = np.full(ts.num_mutations, -1)
    for t in ts.trees():
        for m in t.mutations():
            if m.edge != tskit.NULL:
                freq[m.id] = t.num_samples(m.node)
    doubletons = freq == 2
    doubleton_pos = ts.sites_position[ts.mutations_site][doubletons].astype(int)

    # true ages of doubletons (actual mutation time and edge midpoint), keyed by position
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
    true = np.array([true_ages.get(p, np.nan) for p in doubleton_pos])
    true_mid = np.array([true_midpoint_ages.get(p, np.nan) for p in doubleton_pos])

    # date inferred and (re-date) true trees under each singleton-phasing condition
    infer_stats = {}
    true_stats = {}
    for j, (key, label, _) in enumerate(conditions):
        ts_infer, fit_infer = date_phasing(ts, key, phasing_seed + 2 + j)
        infer_stats[key] = doubleton_stats_by_position(ts_infer, fit_infer, doubleton_pos)
        del ts_infer, fit_infer

        ts_true, fit_true = date_phasing(ts0, key, phasing_seed + 12 + j)
        true_stats[key] = doubleton_stats_by_position(ts_true, fit_true, doubleton_pos)
        del ts_true, fit_true

    ages = {
        "interval_width": interval_width,
        "true": true,
        "true_midpoint": true_mid,
        "infer": infer_stats,
        "true_trees": true_stats,
    }
    pickle.dump(ages, open(cache, "wb"))
else:
    ages = pickle.load(open(cache, "rb"))


interval_width = ages["interval_width"]
truth_actual = ages["true"]
truth_midpoint = ages["true_midpoint"]

def stats(x, y):
    ok = np.logical_and(np.isfinite(x), np.isfinite(y))
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    rmse = np.sqrt(np.mean((x[ok] - y[ok]) ** 2))
    return r, rmse

def coverage_from_pit(pit, widths):
    """
    Central (equal-tailed) interval coverage at each nominal width in `widths`,
    given the probability-integral transform `pit` = P(age <= true age).
    """
    pit = pit[np.isfinite(pit)]
    return np.array([np.mean(((1 - w) / 2 < pit) & (pit < (1 + w) / 2)) for w in widths])

def corrected_pit(d, truth, n_samples=1000, seed=0, chunk=20000):
    """
    P(age <= `truth`) under the exact within-branch model for a doubleton: the age
    is `t_child + U * (t_parent - t_child)`, `U ~ Uniform(0, 1)`. `t_child` and
    `t_parent` are drawn from their (marginal) gamma node posteriors conditioned
    on `t_parent > t_child`, and the uniform position is integrated analytically
    (`F = mean clip((truth - t_child)/(t_parent - t_child)))`). This is an
    independence approximation -- it ignores the child/parent correlation, so it
    slightly over-disperses the branch length. Missing entries are `nan`.
    """
    cm, cv, pm, pv = d["child_mean"], d["child_variance"], d["parent_mean"], d["parent_variance"]
    ok = np.isfinite(cm) & (cv > 0) & np.isfinite(pm) & (pv > 0) & np.isfinite(truth)
    out = np.full(truth.shape, np.nan)
    idx = np.flatnonzero(ok)
    sc, rc = cm[idx] ** 2 / cv[idx], cm[idx] / cv[idx]
    sp, rp = pm[idx] ** 2 / pv[idx], pm[idx] / pv[idx]
    tr = truth[idx]
    rng = np.random.default_rng(seed)
    F = np.full(idx.size, np.nan)
    for s in range(0, idx.size, chunk):
        sl = slice(s, min(s + chunk, idx.size))
        n = sl.stop - sl.start
        tc = rng.gamma(sc[sl, None], 1 / rc[sl, None], size=(n, n_samples))
        tp = rng.gamma(sp[sl, None], 1 / rp[sl, None], size=(n, n_samples))
        valid = tp > tc
        frac = np.clip((tr[sl, None] - tc) / np.where(valid, tp - tc, 1.0), 0, 1)
        F[sl] = np.nanmean(np.where(valid, frac, np.nan), axis=1)
    out[idx] = F
    return out

# --- precompute the sampled coverage curves once and cache them, so the figure (and visual
# tweaks to it) regenerate without re-sampling. Delete this cache to force recomputation.
curves_cache = "../data/phasing_benchmark_doubleton_curves.pkl"
if not os.path.exists(curves_cache):
    curves = {treeset: {key: coverage_from_pit(corrected_pit(ages[treeset][key], truth_actual), interval_width)
                        for key, _, _ in conditions}
              for treeset in ("true_trees", "infer")}
    pickle.dump(curves, open(curves_cache, "wb"))
else:
    curves = pickle.load(open(curves_cache, "rb"))

# --- unified figure: accuracy (top row) and corrected-interval coverage (bottom row).
# A single 2 x N gridspec (rather than fig.subfigures) keeps the panel columns aligned
# across rows: one shared gridspec gives each column a common left/right edge.
plot_path = "../figures/phasing_benchmark_doubleton.pdf"
xm = np.nanmean(truth_midpoint)
xmp = xm + 1
fig, axs = plt.subplots(
    2, len(conditions),
    figsize=(len(conditions) * 2.5, 5.4),
    sharex="row", sharey="row",
    constrained_layout=True,
)
for i, (key, label, panel) in enumerate(conditions):
    # accuracy: estimated vs true midpoint age -- the information limit for the point
    # estimate (the recoverable value is the edge midpoint), inferred trees
    est = ages["infer"][key]["mean"]
    r, rmse = stats(np.log10(truth_midpoint), np.log10(est))
    axs[0, i].hexbin(truth_midpoint, est, mincnt=1, xscale="log", yscale="log")
    axs[0, i].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
    axs[0, i].text(0.01, 0.99, f"{label}\n$r={r:.3f}$", size=9, transform=axs[0, i].transAxes, ha='left', va='top')
    if i == 0:
        axs[0, i].set_title("A.", loc="left", fontweight="bold")
    # coverage: corrected posterior interval coverage of doubleton ages (precomputed/cached)
    cov_true = curves["true_trees"][key]
    cov_infer = curves["infer"][key]
    axs[1, i].plot(interval_width, cov_true, "-o", markersize=2, color="firebrick", label="true+tsdate")
    axs[1, i].plot(interval_width, cov_infer, "-o", markersize=2, color="dodgerblue", label="tsinfer+tsdate")
    axs[1, i].axline((0.5, 0.5), (0.6, 0.6), color="black", linestyle="dashed")
    if i == 0:
        axs[1, i].set_title("B.", loc="left", fontweight="bold")
    axs[1, i].text(0.01, 0.99, label, size=9, transform=axs[1, i].transAxes, ha='left', va='top')
    axs[1, i].set_xlim(0, 1)
    axs[1, i].set_ylim(0, 1)
    axs[1, i].set_xticks([0.3, 0.6, 0.9])

axs[0, 0].set_ylabel("Estimated doubleton age", size=10)
axs[1, 0].set_ylabel("Posterior interval coverage", size=10)
axs[1, 0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=8, frameon=False)

# one centered x-axis label per row, via an invisible axes spanning the row's columns
gs = axs[0, 0].get_gridspec()
for r, xlabel in [(0, "True midpoint age (doubleton mutations)"),
                  (1, "Expected interval coverage (true doubleton age)")]:
    span = fig.add_subplot(gs[r, :])
    span.set_xticks([])
    span.set_yticks([])
    span.set_frame_on(False)
    span.set_xlabel(xlabel, size=10, labelpad=20)
fig.savefig(plot_path)
