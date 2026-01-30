import os
import pickle
import tskit
import numpy as np

import mpmath
import scipy
from scipy.special import betaln
from scipy.special import gammaln
from math import exp
from math import log

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch


OVERWRITE_CACHED_RESULTS = False  # cache stuff so that plots can be tweaked quickly
cache_path = "../data/exact_posteriors.pkl"

matplotlib.rcParams['figure.dpi'] = 300
figure = plt.figure(figsize=(10, 3.25), constrained_layout=True)
pre, update, post  = figure.subfigures(1, 3).ravel()
axs = [pre.subplots(1), update.subplots(1), post.subplots(1)]

# --- haplotype schematic helpers --- #

def draw_edge(axs, xlb, xlt, xrb, xrt, yb, yt, s=0.5, color="blue", alpha=0.1):
    """
    (xlt, yt) ----- (xrt, yt)
        |                |
     wiggle            wiggle
        |                |
    (xlb, yb) ----- (xrb, yb)
    """
    knots = 1000
    ly = np.linspace(-6, 6, knots)
    lx = 1 / (1 + np.exp(-s * ly))
    ly = (ly - ly.min()) / (ly.max() - ly.min()) * (yt - yb) + yb
    lx = (lx - lx.min()) / (lx.max() - lx.min()) * (xlt - xlb) + xlb
    rx, ry = lx + (xrb - xlb), ly.copy()
    bx, by = np.linspace(xlb, xrb, knots), np.repeat(yb, knots)
    tx, ty = np.linspace(xlt, xrt, knots), np.repeat(yt, knots)
    poly_x = np.concatenate([bx, rx, tx[::-1], lx[::-1]])
    poly_y = np.concatenate([by, ry, ty[::-1], ly[::-1]])
    poly = [(x, y) for x, y in zip(poly_x, poly_y)]
    axs.add_patch(Polygon(poly, facecolor=color, fill=True, alpha=alpha))

def draw_haplotype(axs, xl, xr, yb, yt, color="lightgray", alpha=1.0):
    poly_x = [xl, xl, xr, xr, xl]
    poly_y = [yb, yt, yt, yb, yb]
    poly = [(x, y) for x, y in zip(poly_x, poly_y)]
    axs.add_patch(Polygon(poly, facecolor=color, alpha=alpha))

def draw_mutation_parent(axs, sample_x, sample_y, parent_x, parent_y, parent_offset, x_pos=0.5, y_pos=0.5, s=0.5, markersize=6, color="blue"):
    knots = 1000
    y1, y0, ym = parent_y[0], sample_y[1], sample_y[0]
    x0 = (sample_x[1] - sample_x[0]) * x_pos + sample_x[0]
    x1 = x0 + parent_offset
    ly = np.linspace(-6, 6, knots)
    lx = 1 / (1 + np.exp(-s * ly))
    ly = (ly - ly.min()) / (ly.max() - ly.min()) * (y1 - y0) + y0
    lx = (lx - lx.min()) / (lx.max() - lx.min()) * (x1 - x0) + x0
    idx = np.searchsorted(np.arange(knots) / knots, y_pos, side='right')
    ly = ly[:idx]
    lx = lx[:idx]
    axs.plot((x0, x0), (ym, y0), "-", color=color)
    axs.plot(lx, ly, "-", color=color, alpha=0.1)
    axs.plot(lx[-1], ly[-1], "o", markersize=markersize, color=color)

def draw_mutation_root(axs, sample_x, sample_y, parent_x, parent_y, parent_offset, root_x, root_y, root_offset, x_pos=0.5, y_pos=0.5, s=0.5, markersize=6, color="blue"):
    knots = 1000
    # line on sample edge
    y1, y0, ym = parent_y[0], sample_y[1], sample_y[0]
    x0 = (sample_x[1] - sample_x[0]) * x_pos + sample_x[0]
    x1 = x0 + parent_offset
    ly = np.linspace(-6, 6, knots)
    lx = 1 / (1 + np.exp(-s * ly))
    ly = (ly - ly.min()) / (ly.max() - ly.min()) * (y1 - y0) + y0
    lx = (lx - lx.min()) / (lx.max() - lx.min()) * (x1 - x0) + x0
    axs.plot((x0, x0), (ym, y0), "-", color=color, linestyle="-")
    axs.plot(lx, ly, "-", color=color, alpha=0.1)
    # line on parent edge
    y1, y0, ym = root_y[0], parent_y[1], parent_y[0]
    x0 = x1
    x1 = x0 + (root_offset - parent_offset)
    ly = np.linspace(-6, 6, knots)
    lx = 1 / (1 + np.exp(-s * ly))
    ly = (ly - ly.min()) / (ly.max() - ly.min()) * (y1 - y0) + y0
    lx = (lx - lx.min()) / (lx.max() - lx.min()) * (x1 - x0) + x0
    idx = np.searchsorted(np.arange(knots) / knots, y_pos, side='right')
    ly = ly[:idx]
    lx = lx[:idx]
    axs.plot((x0, x0), (ym, y0), "-", color=color)
    axs.plot(lx, ly, "-", color=color, alpha=0.1)
    axs.plot(lx[-1], ly[-1], "o", markersize=markersize, color=color)


# --- EP detail helpers --- #

def moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_i, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j
    """
    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t
    f0 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, z)))
    f1 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 1, c + 1, z)))
    f2 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 2, c + 2, z)))
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * exp(f1 - f0)
    d2 = s2 * exp(f2 - f0)
    logl = f0 + betaln(y_ij + 1, a) + gammaln(b) - b * log(t)
    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2
    mn_i = mn_j * z + b / t
    sq_i = sq_j * z**2 + (b + 1) * (mn_i + mn_j * z) / t
    va_i = sq_i - mn_i**2
    return logl, mn_i, va_i, mn_j, va_j


def approximate_pdf(t_i, t_j, a_i, b_i, a_j, b_j):
    logconst_i = gammaln(a_i) - log(b_i) * a_i
    logconst_j = gammaln(a_j) - log(b_j) * a_j
    return exp(
        log(t_i) * (a_i - 1) +
        -t_i * b_i +
        log(t_j) * (a_j - 1) +
        -t_j * b_j +
        -logconst_i - logconst_j
    )


def exact_pdf(t_i, t_j, a_i, b_i, a_j, b_j, y_ij, mu_ij):
    if t_i <= t_j: return 0.0
    logconst, *_ = moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)
    return exp(
        log(t_i) * (a_i - 1) +
        -t_i * b_i +
        log(t_j) * (a_j - 1) +
        -t_j * b_j +
        log(t_i - t_j) * y_ij +
        -(t_i - t_j) * mu_ij +
        -logconst
    )


# --- true posterior helpers --- #

def rejection_sample(
        num_free_nodes,
        edge_parents, 
        edge_children, 
        edge_likelihoods, 
        proposal=[1, 1], # gamma shape/rate
        num_samples=1000, 
        seed=1,
    ):
    """ Get the true posterior via rejection sampling """
    rng = np.random.default_rng(seed)
    batch_size = int(1e6)
    shape, rate = proposal
    edge_muts, edge_span = edge_likelihoods.T
    num_accept = 0
    output = []
    while num_accept < num_samples:
        # sample from proposal
        proposal = rng.gamma(shape, 1/rate, size=(num_free_nodes, batch_size))
        pg = np.sum( # gamma pdf
            (shape - 1) * np.log(proposal) - rate * proposal + 
            -gammaln(shape) + np.log(rate) * shape, 
            axis=0,
        )
        # unormalised posterior log prob
        proposal = np.vstack([np.zeros(batch_size), proposal]) # sample age
        edge_area = (proposal[edge_parents] - proposal[edge_children]) * edge_span[:, np.newaxis]
        reject = np.any(edge_area <= 0, axis=0)
        fg = np.sum( # poisson pmf
            np.log(edge_area) * edge_muts[:, np.newaxis] - \
                edge_area - gammaln(edge_muts[:, np.newaxis] + 1),
            axis=0,
        )
        fg[reject] = -np.inf
        # acceptance sampling
        const = np.maximum(pg, fg)
        pg, fg = np.exp(pg - const), np.exp(fg - const)
        accept = rng.uniform(0, pg, size=pg.size) < fg
        output.append(proposal.T[accept])
        num_accept += np.sum(accept)
    return np.concatenate(output, axis=0)[:num_samples]


# --- setup

offset = 2
# these coordinates are in the original space, and
# and offset is added for aesthetics
sample_x = [0, 14]
sample_y = [0, 0.5]
lparent_x = [0, 10]
lparent_y = [8.5, 9]
rparent_x = [10, 14]
rparent_y = [8.5, 9]
lroot_x = [0, 3]
lroot_y = [17, 17.5]
rroot_x = [3, 14]
rroot_y = [17, 17.5]

mut_x = [0.1, 0.3, 0.4, 0.6, 0.75, 0.8, 0.84, 0.88, 0.91, 0.93, 0.95, 0.97]
mut_y = [0.4, 1.1, 1.5, 0.2, 0.20, 1.7, 0.40, 0.85, 0.10, 0.80, 0.20, 0.45]

# --- DATA FOR PANEL 2: make a tree sequence and date it
if not os.path.exists(cache_path) or OVERWRITE_CACHED_RESULTS:

    tables = tskit.TableCollection(sequence_length=sample_x[-1])
    tables.nodes.add_row(time=sample_y[0], flags=tskit.NODE_IS_SAMPLE)
    tables.nodes.add_row(time=lparent_y[0], flags=0)
    tables.nodes.add_row(time=rparent_y[0], flags=0)
    tables.nodes.add_row(time=lroot_y[0], flags=0)
    tables.nodes.add_row(time=rroot_y[0], flags=0)
    tables.edges.add_row(left=lparent_x[0], right=lparent_x[1], parent=1, child=0)
    tables.edges.add_row(left=rparent_x[0], right=rparent_x[1], parent=2, child=0)
    tables.edges.add_row(left=lroot_x[0], right=lroot_x[1], parent=3, child=1)
    tables.edges.add_row(left=rroot_x[0], right=lparent_x[1], parent=4, child=1)
    tables.edges.add_row(left=rparent_x[0], right=rroot_x[1], parent=4, child=2)
    for xp, yp in zip(mut_x, mut_y):
        x = xp * sample_x[1]
        if yp > 1: # upper edges
            node = 1 if x < lparent_x[1] else 2
        else: # lower edges
            node = 0
        site = tables.sites.add_row(position=x, ancestral_state="-")
        tables.mutations.add_row(site=site, node=node, time=tskit.UNKNOWN_TIME, derived_state="*")
    ts = tables.tree_sequence()
    
    import tsdate
    ep = tsdate.variational.ExpectationPropagation(ts, mutation_rate=1.0, allow_unary=True)
    # run up to last edge update on upward pass
    ep.propagate_likelihood(
        ep.edge_order[:4],
        ep.edge_parents,
        ep.edge_children,
        ep.edge_likelihoods,
        ep.node_constraints,
        ep.node_posterior,
        ep.edge_factors,
        ep.edge_logconst,
        ep.node_scale,
        1000,
        0.1,
        tsdate.variational.USE_EDGE_LIKELIHOOD,
    )

    # save state for contour plot
    edge = ep.edge_order[4]
    child = ep.edge_children[edge]
    parent = ep.edge_parents[edge]
    child_cavity = ep.node_posterior[child].copy()
    parent_cavity = ep.node_posterior[parent].copy()
    likelihood = ep.edge_likelihoods[edge].copy()

    # do final edge update
    ep.propagate_likelihood(
        ep.edge_order[[4]],
        ep.edge_parents,
        ep.edge_children,
        ep.edge_likelihoods,
        ep.node_constraints,
        ep.node_posterior,
        ep.edge_factors,
        ep.edge_logconst,
        ep.node_scale,
        1000,
        0.1,
        tsdate.variational.USE_EDGE_LIKELIHOOD,
    )
    mean, var = ep.node_moments()

    # now run until convergence to compare with true posterior
    ep_posteriors = [ep.node_posterior.copy()]
    for k in range(20): 
        ep.propagate_likelihood(
            ep.edge_order[:5],
            ep.edge_parents,
            ep.edge_children,
            ep.edge_likelihoods,
            ep.node_constraints,
            ep.node_posterior,
            ep.edge_factors,
            ep.edge_logconst,
            ep.node_scale,
            1000,
            0.1,
            tsdate.variational.USE_EDGE_LIKELIHOOD,
        )
        ep_posteriors.append(ep.node_posterior.copy())
    final_mean, final_var = ep.node_moments()

    # now rejection sample to get true posterior
    # (this is the costly part)
    true_posteriors = \
        rejection_sample(
            ts.nodes_time.size - 1, 
            ep.edge_parents, 
            ep.edge_children, 
            ep.edge_likelihoods,
            proposal=[1., 1.], # for efficiency this has to match timescale
            num_samples=10000,
        ).T

    # save this stuff
    pickle.dump(
        {
            "mean" : mean,
            "parent" : parent,
            "child" : child,
            "child_cavity" : child_cavity,
            "parent_cavity" : parent_cavity,
            "likelihood" : likelihood,
            "ep_posteriors" : ep_posteriors,
            "true_posteriors" : true_posteriors,
        },
        open(cache_path, "wb"),
    )
else:
    p2_cache = pickle.load(open(cache_path, "rb"))
    mean = p2_cache["mean"]
    parent = p2_cache["parent"]
    child = p2_cache["child"]
    child_cavity = p2_cache["child_cavity"]
    parent_cavity = p2_cache["parent_cavity"]
    likelihood = p2_cache["likelihood"]
    ep_posteriors = p2_cache["ep_posteriors"]
    true_posteriors = p2_cache["true_posteriors"]


# --- PANEL 1: dated tree sequence prior to update --- #

# ages before update
pre_mean = np.array(mean.copy())
pre_mean[parent] = (parent_cavity[0] + 1) / parent_cavity[1]
pre_mean[child] = (child_cavity[0] + 1) / child_cavity[1]

scaling = max(lroot_y[1], rroot_y[1]) / pre_mean.max()
nodes_time = pre_mean * scaling
sample_y = [0, 0.5]
lparent_y = [nodes_time[1], nodes_time[1] + 0.5]
rparent_y = [nodes_time[2], nodes_time[2] + 0.5]
lroot_y = [nodes_time[3], nodes_time[3] + 0.5]
rroot_y = [nodes_time[4], nodes_time[4] + 0.5]
draw_edge(axs[0], lparent_x[0], lparent_x[0] - offset, lparent_x[1], lparent_x[1] - offset, sample_y[1], lparent_y[0], s=0.5) # sample -> left parent
draw_edge(axs[0], rparent_x[0], rparent_x[0] + offset, rparent_x[1], rparent_x[1] + offset, sample_y[1], rparent_y[0], s=0.5) # sample -> right parent
draw_edge(axs[0], lroot_x[0] - offset, lroot_x[0] - 2*offset, lroot_x[1] - offset, lroot_x[1] - 2*offset, lparent_y[1], lroot_y[0], s=0.5) # left parent -> left root
draw_edge(axs[0], rroot_x[0] - offset, rroot_x[0], lparent_x[1] - offset, rroot_x[1], lparent_y[1], rroot_y[0], s=0.5) # left parent -> right root

#draw_edge(axs[0], rparent_x[0] + offset, rparent_x[0], rparent_x[1] + offset, rroot_x[1], rparent_y[1], rroot_y[0], s=0.5, color="firebrick", alpha=0.2) # right parent -> right root #<<< with highlight
#draw_edge(axs[0], rparent_x[0] + offset, rparent_x[0], rparent_x[1] + offset, rroot_x[1], rparent_y[1], rroot_y[0], s=0.5, alpha=0.1) # right parent -> right root #<<< without highlight

draw_haplotype(axs[0], sample_x[0], sample_x[1], sample_y[0], sample_y[1]) # sample
draw_haplotype(axs[0], lparent_x[0] - offset, lparent_x[1] - offset, lparent_y[0], lparent_y[1]) # left sample parent
draw_haplotype(axs[0], rparent_x[0] + offset, rparent_x[1] + offset, rparent_y[0], rparent_y[1]) # right sample parent
draw_haplotype(axs[0], lroot_x[0] - offset * 2, lroot_x[1] - offset * 2, lroot_y[0], lroot_y[1]) # left root
#draw_haplotype(axs[0], rroot_x[0] - offset * 0, rroot_x[1] - offset * 0, rroot_y[0], rroot_y[1]) # left root
draw_haplotype(axs[0], rroot_x[0] - offset * 0, lparent_x[1] - offset * 0, rroot_y[0], rroot_y[1]) # left root

for xp, yp in zip(mut_x, mut_y):
    if yp > 1: # upper edges
        if xp < lroot_x[1] / sample_x[1]:
            draw_mutation_root(axs[0], sample_x, sample_y, lparent_x, lparent_y, -offset, lroot_x, lroot_y, -2*offset, s=0.5, x_pos=xp, y_pos=yp-1, markersize=3)
        elif xp > lroot_x[1] / sample_x[1] and xp < rparent_x[0] / sample_x[1]:
            draw_mutation_root(axs[0], sample_x, sample_y, lparent_x, lparent_y, -offset, rroot_x, rroot_y, 0, s=0.5, x_pos=xp, y_pos=yp-1, markersize=3)
        #else:
        #    draw_mutation_root(axs[0], sample_x, sample_y, rparent_x, rparent_y, offset, rroot_x, rroot_y, 0, s=0.5, x_pos=xp, y_pos=yp-1, markersize=3)
    else: # lower edges
        if xp < lparent_x[1] / sample_x[1]:
            draw_mutation_parent(axs[0], sample_x, sample_y, lparent_x, lparent_y, -offset, s=0.5, x_pos=xp, y_pos=yp, markersize=3)
        else:
            draw_mutation_parent(axs[0], sample_x, sample_y, rparent_x, rparent_y, offset, s=0.5, x_pos=xp, y_pos=yp, markersize=3)

axs[0].set_title("A", ha="right", x=-0.13, size=14)
axs[0].set_xlim(sample_x[0] - 3 * offset, sample_x[1] + 2 * offset)
axs[0].set_ylim(sample_y[0] - 0.1 * offset, max(rroot_y[1], lroot_y[1], rparent_y[1], lparent_y[1]) + offset)
axs[0].set_ylabel("Generations in past")
axs[0].set_yticks(np.array([0, 0.8, 1.6]) * scaling)
axs[0].set_yticklabels(np.array([0, 0.8, 1.6]))
axs[0].set_xlabel(" ")
axs[0].set_xticklabels([" ", " "])
axs[0].text(-6.25, 2.25 * scaling, r"$\times 10^3$", va="top", ha="right", fontsize=9)
axs[0].tick_params(axis="x", color="white")
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].text(0.03, 0.99, r"Before edge $b \rightarrow d$", ha="left", va="top", size=12, transform=axs[0].transAxes)
axs[0].text(sample_x[0]/2 + sample_x[1]/2, sample_y[1] - 0.75, "sample haplotype", ha="center", va="top", color="gray")

nl = 0.2
axs[0].text(lparent_x[0] - offset - nl, lparent_y[0], "a", ha="right", color="gray", size=10)
axs[0].text(lroot_x[0] - offset * 2 - nl, lroot_y[0], "c", ha="right", color="gray", size=10)
axs[0].text(rparent_x[1] + offset + nl, rparent_y[0], "b", ha="left", color="gray", size=10)
axs[0].text(rroot_x[0] - nl, rroot_y[0], "d", ha="right", color="gray", size=10)


# --- DATA FOR PANEL 3: calculate log posteriors --- #
a_i = parent_cavity[0] + 1
b_i = parent_cavity[1]
a_j = child_cavity[0] + 1
b_j = child_cavity[1]
y_ij = likelihood[0]
mu_ij = likelihood[1]

_, mn_i, va_i, mn_j, va_j = moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)
au_i, bu_i = mn_i ** 2 / va_i, mn_i / va_i
au_j, bu_j = mn_j ** 2 / va_j, mn_j / va_j

t_max = max(mean[parent] * 2, mean[child] * 2)
grid_i = np.linspace(0, t_max, 101)[1:]
grid_j = np.linspace(0, t_max, 101)[1:]
logl_cavity = np.zeros((grid_i.size, grid_j.size))
logl_exact = np.zeros((grid_i.size, grid_j.size))
logl_update = np.zeros((grid_i.size, grid_j.size))

for i, t_i in enumerate(grid_i):
    for j, t_j in enumerate(grid_j):
        logl_cavity[i, j] = approximate_pdf(t_i, t_j, a_i, b_i, a_j, b_j)
        logl_exact[i, j] = exact_pdf(t_i, t_j, a_i, b_i, a_j, b_j, y_ij, mu_ij)
        logl_update[i, j] = approximate_pdf(t_i, t_j, au_i, bu_i, au_j, bu_j)
logl_exact[logl_exact == 0] = logl_exact[logl_exact > 0].min()



def quantile_to_level(data, quantile):
    """Return data levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels


# --- PANEL 2: EP detail as contour plot --- #

levels = np.linspace(0.2, 1.0, 10) # probability isoclines

poly = [(0, 0), (t_max, t_max), (t_max, 0), (0, 0)]
axs[1].add_patch(Polygon(poly, facecolor="lightgray", fill=True, alpha=0.5))
axs[1].contour(
    grid_i, grid_j, logl_cavity, 
    levels=quantile_to_level(logl_cavity, levels), 
    colors="dodgerblue",
)
axs[1].text(1.0, 0.1, "Posterior without edge", ha="left", va="center", color="dodgerblue", size=10)
axs[1].contour(
    grid_i, grid_j, logl_exact, 
    levels=quantile_to_level(logl_exact, levels), 
    colors="black",
)
axs[1].text(0.05, 0.5, "Edge update", ha="left", va="center", color="black", size=10)
axs[1].contour(
    grid_i, grid_j, logl_update, 
    levels=quantile_to_level(logl_update, levels), 
    colors="firebrick",
)
axs[1].text(0.5, 1.9, "Match moments", ha="left", va="bottom", color="firebrick", size=10)
axs[1].add_patch(
    FancyArrowPatch((1.0, 0.1), (0.28, 0.45), mutation_scale=10,
        connectionstyle="arc3,rad=-.25", arrowstyle=u"->", lw=1)
)
axs[1].add_patch(
    FancyArrowPatch((0.25, 0.55), (0.5, 1.89), mutation_scale=10,
        connectionstyle="arc3,rad=-.25", arrowstyle=u"->", lw=1)
)
axs[1].set_title("B", ha="right", x=-0.13, size=14)
axs[1].set_xlim(grid_j.min(), grid_j.max())
axs[1].set_ylim(grid_i.min(), grid_i.max())
axs[1].set_ylabel("Parent age (node d)")
axs[1].set_xlabel("Child age (node b)")
axs[1].text(2.48, -0.02, r"$\times 10^3$", va="top", ha="right", fontsize=9)
axs[1].text(-0.005, 2.45, r"$\times 10^3$", va="top", ha="right", fontsize=9)
axs[1].text(0.03, 0.99, "Expectation propagation", transform=axs[1].transAxes, ha="left", va="top", color="black", size=12)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)


# --- PANEL 3: dated tree sequence after to update --- #

# setup again
offset = 2
# these coordinates are in the original space, and
# and offset is added for aesthetics
sample_x = [0, 14]
sample_y = [0, 0.5]
lparent_x = [0, 10]
lparent_y = [8.5, 9]
rparent_x = [10, 14]
rparent_y = [8.5, 9]
lroot_x = [0, 3]
lroot_y = [17, 17.5]
rroot_x = [3, 14]
rroot_y = [17, 17.5]
mut_x = [0.1, 0.3, 0.4, 0.6, 0.75, 0.8, 0.84, 0.88, 0.91, 0.93, 0.95, 0.97]
mut_y = [0.4, 1.1, 1.5, 0.2, 0.20, 1.7, 0.40, 0.85, 0.10, 0.80, 0.20, 0.45]

# ages after update
post_mean = mean.copy()

scaling = max(lroot_y[1], rroot_y[1]) / post_mean.max()
nodes_time = post_mean * scaling
sample_y = [0, 0.5]
lparent_y = [nodes_time[1], nodes_time[1] + 0.5]
rparent_y = [nodes_time[2], nodes_time[2] + 0.5]
lroot_y = [nodes_time[3], nodes_time[3] + 0.5]
rroot_y = [nodes_time[4], nodes_time[4] + 0.5]
draw_edge(axs[2], lparent_x[0], lparent_x[0] - offset, lparent_x[1], lparent_x[1] - offset, sample_y[1], lparent_y[0], s=0.5) # sample -> left parent
draw_edge(axs[2], rparent_x[0], rparent_x[0] + offset, rparent_x[1], rparent_x[1] + offset, sample_y[1], rparent_y[0], s=0.5) # sample -> right parent
draw_edge(axs[2], lroot_x[0] - offset, lroot_x[0] - 2*offset, lroot_x[1] - offset, lroot_x[1] - 2*offset, lparent_y[1], lroot_y[0], s=0.5) # left parent -> left root
draw_edge(axs[2], rroot_x[0] - offset, rroot_x[0], lparent_x[1] - offset, rroot_x[1], lparent_y[1], rroot_y[0], s=0.5) # left parent -> right root
draw_edge(axs[2], rparent_x[0] + offset, rparent_x[0], rparent_x[1] + offset, rroot_x[1], rparent_y[1], rroot_y[0], s=0.5, alpha=0.1) # right parent -> right root 

draw_haplotype(axs[2], sample_x[0], sample_x[1], sample_y[0], sample_y[1]) # sample
draw_haplotype(axs[2], lparent_x[0] - offset, lparent_x[1] - offset, lparent_y[0], lparent_y[1]) # left sample parent
draw_haplotype(axs[2], rparent_x[0] + offset, rparent_x[1] + offset, rparent_y[0], rparent_y[1]) # right sample parent
draw_haplotype(axs[2], lroot_x[0] - offset * 2, lroot_x[1] - offset * 2, lroot_y[0], lroot_y[1]) # left root
draw_haplotype(axs[2], rroot_x[0] - offset * 0, rroot_x[1] - offset * 0, rroot_y[0], rroot_y[1]) # left root

for xp, yp in zip(mut_x, mut_y):
    if yp > 1: # upper edges
        if xp < lroot_x[1] / sample_x[1]:
            draw_mutation_root(axs[2], sample_x, sample_y, lparent_x, lparent_y, -offset, lroot_x, lroot_y, -2*offset, s=0.5, x_pos=xp, y_pos=yp-1, markersize=3)
        elif xp > lroot_x[1] / sample_x[1] and xp < rparent_x[0] / sample_x[1]:
            draw_mutation_root(axs[2], sample_x, sample_y, lparent_x, lparent_y, -offset, rroot_x, rroot_y, 0, s=0.5, x_pos=xp, y_pos=yp-1, markersize=3)
        else:
            draw_mutation_root(axs[2], sample_x, sample_y, rparent_x, rparent_y, offset, rroot_x, rroot_y, 0, s=0.5, x_pos=xp, y_pos=yp-1, markersize=3)
    else: # lower edges
        if xp < lparent_x[1] / sample_x[1]:
            draw_mutation_parent(axs[2], sample_x, sample_y, lparent_x, lparent_y, -offset, s=0.5, x_pos=xp, y_pos=yp, markersize=3)
        else:
            draw_mutation_parent(axs[2], sample_x, sample_y, rparent_x, rparent_y, offset, s=0.5, x_pos=xp, y_pos=yp, markersize=3)

axs[2].set_title("C", ha="right", x=-0.13, size=14)
axs[2].set_xlim(sample_x[0] - 3 * offset, sample_x[1] + 2 * offset)
axs[2].set_ylim(sample_y[0] - 0.1 * offset, max(rroot_y[1], lroot_y[1]) + offset)
axs[2].set_ylabel("Generations in past")
axs[2].set_yticks(np.array([0, 0.5, 1]) * scaling)
axs[2].set_yticklabels(np.array([0, 0.5, 1]))
axs[2].set_xlabel(" ")
axs[2].set_xticklabels([" ", " "])
axs[2].text(-6.25, 1.4 * scaling, r"$\times 10^3$", va="top", ha="right", fontsize=9)
axs[2].tick_params(axis="x", color="white")
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['bottom'].set_visible(False)
axs[2].text(0.03, 0.99, r"After edge $b \rightarrow d$", ha="left", va="top", size=12, transform=axs[2].transAxes)
axs[2].text(sample_x[0]/2 + sample_x[1]/2, sample_y[1] - 0.75, "sample haplotype", ha="center", va="top", color="gray")

nl = 0.2
axs[2].text(lparent_x[0] - offset - nl, lparent_y[0], "a", ha="right", color="gray", size=10)
axs[2].text(lroot_x[0] - offset * 2 - nl, lroot_y[0], "c", ha="right", color="gray", size=10)
axs[2].text(rparent_x[1] + offset + nl, rparent_y[0], "b", ha="left", color="gray", size=10)
axs[2].text(rroot_x[1] + nl, rroot_y[0], "d", ha="left", color="gray", size=10)


plt.savefig("../figures/algorithm_schematic_supp.pdf")


