"""
Show the impact of incorrectly assigning phase to singleton mutations on the
*uncertainty* (posterior interval coverage) of singleton age estimates. For
each phasing condition we date both the inferred trees and the true trees
(re-dated under the same phasing condition), so that each panel separates the
cost of tree inference from the cost of mis-phasing.
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
cache = "../data/phasing_benchmark_singletons.pkl"

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


# the phasing conditions shown in the figure (key, panel label, panel letter)
conditions = [
    ("known", "Known phase", "A."),
    ("random", "Random phase", "B."),
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

def singleton_stats_by_position(ts, fit, positions, nodes_true_time=None):
    """
    For each integer position in `positions`, extract from the biallelic sites of
    the dated `ts`/`fit`: the mutation-age posterior (`mean`, `variance`); the
    posterior of the parent node bounding the mutation's branch (`parent_mean`,
    `parent_variance`); the posterior of the parent node above the *other* genome
    of the same (diploid) individual (`alt_parent_mean`, `alt_parent_variance`),
    needed to sample the phase-agnostic two-haplotype mixture; and the chosen
    parent node's true age (`parent_true_time`, from `nodes_true_time` if given
    else `nan`). Entries with no matching (biallelic) site are `nan`.
    """
    mpost = fit.mutation_posteriors()
    npost = fit.node_posteriors()
    muts_per_site = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    keys = ["mean", "variance", "parent_mean", "parent_variance",
            "alt_parent_mean", "alt_parent_variance", "parent_true_time"]
    record = {}
    for tree in ts.trees():
        for m in tree.mutations():
            if muts_per_site[m.site] != 1 or m.edge == tskit.NULL:
                continue
            parent = tree.parent(m.node)
            if parent == tskit.NULL:
                continue
            # parent node above the individual's other genome (the phase alternative)
            alt_mean = alt_var = np.nan
            individual = ts.nodes_individual[m.node]
            if individual != tskit.NULL:
                others = [n for n in ts.individual(individual).nodes if n != m.node]
                if others:
                    alt_parent = tree.parent(others[0])
                    if alt_parent != tskit.NULL:
                        alt_mean, alt_var = npost["mean"][alt_parent], npost["variance"][alt_parent]
            pos = int(ts.sites_position[m.site])
            record[pos] = (
                mpost["mean"][m.id], mpost["variance"][m.id],
                npost["mean"][parent], npost["variance"][parent],
                alt_mean, alt_var,
                np.nan if nodes_true_time is None else nodes_true_time[parent],
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

    # singletons (defined on the inferred trees) and their genomic positions
    freq = np.full(ts.num_mutations, -1)
    for t in ts.trees():
        for m in t.mutations():
            if m.edge != tskit.NULL:
                freq[m.id] = t.num_samples(m.node)
    singletons = freq == 1
    singleton_pos = ts.sites_position[ts.mutations_site][singletons].astype(int)

    # true ages of singletons (actual mutation time and edge midpoint), keyed by position
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
    true = np.array([true_ages.get(p, np.nan) for p in singleton_pos])
    true_mid = np.array([true_midpoint_ages.get(p, np.nan) for p in singleton_pos])

    # date inferred and (re-date) true trees under each phasing condition
    infer_stats = {}
    true_stats = {}
    for j, (key, label, _) in enumerate(conditions):
        ts_infer, fit_infer = date_phasing(ts, key, phasing_seed + 2 + j)
        infer_stats[key] = singleton_stats_by_position(ts_infer, fit_infer, singleton_pos)
        del ts_infer, fit_infer

        ts_true, fit_true = date_phasing(ts0, key, phasing_seed + 12 + j)
        true_stats[key] = singleton_stats_by_position(
            ts_true, fit_true, singleton_pos, nodes_true_time=ts0.nodes_time,
        )
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


# --- make figure
def coverage_from_pit(pit, widths):
    """
    Central (equal-tailed) interval coverage at each nominal width in `widths`,
    given the probability-integral transform `pit` = P(age <= true age). For a
    calibrated posterior `pit` is uniform, so coverage equals the fraction of
    `pit` within the central interval of each width.
    """
    pit = pit[np.isfinite(pit)]
    return np.array([np.mean(((1 - w) / 2 < pit) & (pit < (1 + w) / 2)) for w in widths])

def sampled_pit(d, truth, marginalize_phase, n_samples=1000, seed=0, chunk=20000):
    """
    P(age <= `truth`) for a singleton, by sampling the exact within-branch posterior
    given the node gamma posteriors: the age is `U * t_parent`, `U ~ Uniform(0, 1)`,
    with `t_parent` drawn from the parent node's gamma posterior (the uniform
    position is integrated analytically, `F = mean clip(truth / t_parent)`). When
    `marginalize_phase` is True (the phase-agnostic algorithm), the genome carrying
    the singleton is itself uncertain, so `t_parent` is drawn from the two-haplotype
    mixture: pick a genome with probability proportional to its branch length, then
    use that genome's parent. Missing entries are `nan`.
    """
    cm, cv = d["parent_mean"], d["parent_variance"]
    ok = np.isfinite(cm) & (cv > 0) & np.isfinite(truth)
    if marginalize_phase:
        am, av = d["alt_parent_mean"], d["alt_parent_variance"]
        ok = ok & np.isfinite(am) & (av > 0)
    out = np.full(truth.shape, np.nan)
    idx = np.flatnonzero(ok)
    sc, rc = cm[idx] ** 2 / cv[idx], cm[idx] / cv[idx]
    tr = truth[idx]
    rng = np.random.default_rng(seed)
    if marginalize_phase:
        sa, ra = am[idx] ** 2 / av[idx], am[idx] / av[idx]
    F = np.full(idx.size, np.nan)
    for s in range(0, idx.size, chunk):
        sl = slice(s, min(s + chunk, idx.size))
        n = sl.stop - sl.start
        t_chosen = rng.gamma(sc[sl, None], 1 / rc[sl, None], size=(n, n_samples))
        if marginalize_phase:
            t_alt = rng.gamma(sa[sl, None], 1 / ra[sl, None], size=(n, n_samples))
            on_chosen = rng.random((n, n_samples)) < t_chosen / (t_chosen + t_alt)
            t_parent = np.where(on_chosen, t_chosen, t_alt)
        else:
            t_parent = t_chosen
        F[sl] = np.mean(np.clip(tr[sl, None] / t_parent, 0, 1), axis=1)
    out[idx] = F
    return out

interval_width = ages["interval_width"]
truth_actual = ages["true"]
truth_midpoint = ages["true_midpoint"]

def stats(x, y):
    ok = np.logical_and(np.isfinite(x), np.isfinite(y))
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    rmse = np.sqrt(np.mean((x[ok] - y[ok]) ** 2))
    return r, rmse

def width_over_mean(d, widths, marginalize_phase=False, n_samples=2000, seed=0, chunk=20000):
    """
    Median over singletons of the central credible-interval width (at each nominal width in
    `widths`) divided by the posterior mean. Ages are sampled as `U * t_parent`,
    `U ~ Uniform(0, 1)`, with `t_parent` drawn from the parent node's gamma posterior
    (two-haplotype mixture if `marginalize_phase`). The within-branch floor, Uniform(0, T),
    has width/mean = 2w independent of T.
    """
    lo_q, hi_q = (1 - widths) / 2, (1 + widths) / 2
    cm, cv = d["parent_mean"], d["parent_variance"]
    ok = np.isfinite(cm) & (cv > 0)
    if marginalize_phase:
        am, av = d["alt_parent_mean"], d["alt_parent_variance"]
        ok = ok & np.isfinite(am) & (av > 0)
    idx = np.flatnonzero(ok)
    sc, rc = cm[idx] ** 2 / cv[idx], cm[idx] / cv[idx]
    if marginalize_phase:
        sa, ra = am[idx] ** 2 / av[idx], am[idx] / av[idx]
    rng = np.random.default_rng(seed)
    rel = np.full((widths.size, idx.size), np.nan)
    for s in range(0, idx.size, chunk):
        sl = slice(s, min(s + chunk, idx.size))
        n = sl.stop - sl.start
        t_parent = rng.gamma(sc[sl, None], 1 / rc[sl, None], size=(n, n_samples))
        if marginalize_phase:
            t_alt = rng.gamma(sa[sl, None], 1 / ra[sl, None], size=(n, n_samples))
            on_chosen = rng.random((n, n_samples)) < t_parent / (t_parent + t_alt)
            t_parent = np.where(on_chosen, t_parent, t_alt)
        tm = rng.random((n, n_samples)) * t_parent
        width = np.quantile(tm, hi_q, axis=1) - np.quantile(tm, lo_q, axis=1)
        rel[:, sl] = width / tm.mean(axis=1)[None, :]
    return np.nanmedian(rel, axis=1)

# --- precompute the sampled coverage + interval-width curves once and cache them, so the
# figure (and visual tweaks to it) regenerate without re-sampling. Delete this cache to
# force recomputation if the sampling logic changes.
curves_cache = "../data/phasing_benchmark_singletons_curves.pkl"
if not os.path.exists(curves_cache):
    curves = {}
    for treeset in ("true_trees", "infer"):
        curves[treeset] = {}
        for key, _, _ in conditions:
            marg = key == "no_phase"
            d = ages[treeset][key]
            curves[treeset][key] = {
                "coverage": coverage_from_pit(sampled_pit(d, truth_actual, marg), interval_width),
                "width": width_over_mean(d, interval_width, marg),
            }
    pickle.dump(curves, open(curves_cache, "wb"))
else:
    curves = pickle.load(open(curves_cache, "rb"))

# --- combined figure: singleton age accuracy (top row), interval coverage (middle row), and
# interval width (bottom row), with the phasing conditions as columns. A single 3 x N
# gridspec keeps the panel columns aligned across rows.
plot_path = "../figures/phasing_benchmark_singletons.pdf"
xm = np.nanmean(truth_midpoint)
xmp = xm + 1
fig, axs = plt.subplots(
    3, len(conditions),
    figsize=(len(conditions) * 2.5, 8.0),
    sharex="row", sharey="row",
    constrained_layout=True,
)
for i, (key, label, panel) in enumerate(conditions):
    # accuracy: estimated vs true midpoint age -- the information limit for unphased
    # singletons (the recoverable point estimate is the edge midpoint), inferred trees
    est = ages["infer"][key]["mean"]
    r, rmse = stats(np.log10(truth_midpoint), np.log10(est))
    axs[0, i].hexbin(truth_midpoint, est, mincnt=1, xscale="log", yscale="log")
    axs[0, i].axline((xm, xm), (xmp, xmp), linestyle="dashed", color="red")
    axs[0, i].text(0.01, 0.99, f"{label}\n$r={r:.3f}$", size=9, transform=axs[0, i].transAxes, ha='left', va='top')
    if i == 0:
        axs[0, i].set_title("A.", loc="left", fontweight="bold")
    # coverage (sampled interval coverage; precomputed/cached)
    cov_true = curves["true_trees"][key]["coverage"]
    cov_infer = curves["infer"][key]["coverage"]
    axs[1, i].axline((0.5, 0.5), (0.6, 0.6), color="black", linestyle="dashed", label="true")
    axs[1, i].plot(interval_width, cov_true, "-o", markersize=2, color="firebrick", label="true+tsdate")
    axs[1, i].plot(interval_width, cov_infer, "-o", markersize=2, color="dodgerblue", label="tsinfer+tsdate")
    if i == 0:
        axs[1, i].set_title("B.", loc="left", fontweight="bold")
    axs[1, i].text(0.01, 0.99, label, size=9, transform=axs[1, i].transAxes, ha='left', va='top')
    axs[1, i].set_xlim(0, 1)
    axs[1, i].set_ylim(0, 1)
    axs[1, i].set_xticks([0.3, 0.6, 0.9])
    # interval width relative to the posterior mean (true = 2w, the within-branch floor; cached)
    w_true = curves["true_trees"][key]["width"]
    w_infer = curves["infer"][key]["width"]
    axs[2, i].plot(interval_width, 2 * interval_width, linestyle="dashed", color="black", lw=1.2, label="true")
    axs[2, i].plot(interval_width, w_true, "-o", markersize=2, color="firebrick", label="true+tsdate")
    axs[2, i].plot(interval_width, w_infer, "-o", markersize=2, color="dodgerblue", label="tsinfer+tsdate")
    if i == 0:
        axs[2, i].set_title("C.", loc="left", fontweight="bold")
    axs[2, i].text(0.01, 0.99, label, size=9, transform=axs[2, i].transAxes, ha='left', va='top')
    axs[2, i].set_xlim(0, 1)
    axs[2, i].set_ylim(bottom=0)
    axs[2, i].set_xticks([0.3, 0.6, 0.9])

axs[0, 0].set_ylabel("Estimated singleton age", size=10)
axs[1, 0].set_ylabel("Posterior interval coverage", size=10)
axs[2, 0].set_ylabel("Interval width / posterior mean", size=10)
axs[1, 0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=8, frameon=False)
axs[2, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=8, frameon=False)

# one centered x-axis label per row, via an invisible axes spanning the row's columns
gs = axs[0, 0].get_gridspec()
for r, xlabel in [(0, "True midpoint age (unphased singleton mutations)"),
                  (1, "Expected interval coverage (true singleton age)"),
                  (2, "Expected interval coverage (true singleton age)")]:
    span = fig.add_subplot(gs[r, :])
    span.set_xticks([])
    span.set_yticks([])
    span.set_frame_on(False)
    span.set_xlabel(xlabel, size=10, labelpad=20)
fig.savefig(plot_path)


# --- pedagogical figure: gamma projection vs exact posterior for example singletons
density_plot_path = "../figures/singleton_density_gamma_vs_exact.pdf"

def singleton_density(parent_mean, parent_variance, x):
    """
    Exact posterior density of a singleton's age `U * t_parent`, with
    `U ~ Uniform(0, 1)` (position along the branch above a sample) and `t_parent`
    the parent node's gamma posterior: `f(x) = rate / (shape - 1) *
    P(Gamma(shape - 1) > x)` (valid for parent shape > 1).
    """
    shape, rate = parent_mean ** 2 / parent_variance, parent_mean / parent_variance
    return rate / (shape - 1) * scipy.stats.gamma.sf(x, shape - 1, scale=1 / rate)

# a single illustrative singleton, shown under true-tree and inferred-tree dating
def density_valid(d):
    return (
        np.isfinite(d["parent_mean"]) & (d["parent_variance"] > 0)
        & np.isfinite(d["mean"]) & (d["variance"] > 0)
        & (d["parent_mean"] ** 2 / d["parent_variance"] > 1)
    )

panels = [(ages["true_trees"]["known"], "true+tsdate"),
          (ages["infer"]["known"], "tsinfer+tsdate")]
shared = density_valid(panels[0][0]) & density_valid(panels[1][0]) & np.isfinite(truth_actual)
order = np.flatnonzero(shared)[np.argsort(panels[0][0]["parent_mean"][shared])]
i = order[int(0.7 * order.size)]

dfig, daxs = plt.subplots(1, 2, figsize=(7, 3), sharex=True, constrained_layout=True)
xmax = max(
    max(scipy.stats.gamma.ppf(0.999, d["mean"][i] ** 2 / d["variance"][i], scale=d["variance"][i] / d["mean"][i]),
        d["parent_mean"][i] * 1.25)
    for d, _ in panels
)
x = np.linspace(1e-6, xmax, 800)
for ax, (d, name) in zip(daxs, panels):
    pm, pv, mm, mv = d["parent_mean"][i], d["parent_variance"][i], d["mean"][i], d["variance"][i]
    shape_g, rate_g = mm ** 2 / mv, mm / mv   # the gamma tsdate reports
    ax.plot(x, singleton_density(pm, pv, x), color="dodgerblue", lw=1.8, label="exact convolution")
    ax.plot(x, scipy.stats.gamma.pdf(x, shape_g, scale=1 / rate_g), color="crimson", lw=1.8, label="gamma projection")
    ax.axvline(mm, color="black", linestyle="dashed", lw=1.2, label="posterior mean")
    ax.axvline(truth_actual[i], color="black", linestyle="solid", lw=1.4, label="true age")
    ax.set_title(name, size=10)
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
daxs[0].set_ylabel("Posterior density")
daxs[0].legend(fontsize=9, frameon=False, loc="upper right")
dfig.supxlabel("Singleton age (generations)", size=10)
dfig.savefig(density_plot_path)
