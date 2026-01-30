"""
Code to investigate time-rescaling strategies on trees with artefactual polytomies
"""
import tskit
import msprime
import tszip
import os
import pickle
import numpy as np
import itertools
import collections
import logging

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


# --- code to collapse unsupported edges into polytomies 
# (from https://github.com/tskit-dev/tskit/discussions/2926)

def remove_edges(ts, edge_id_remove_list):
    edges_to_remove_by_child = collections.defaultdict(list)
    edge_id_remove_list = set(edge_id_remove_list)
    for m in ts.mutations():
        if m.edge in edge_id_remove_list:
            # If we remove this edge, we will remove the associated mutation
            # as the child node won't have ancestral material in this region.
            # So we force the user to explicitly (re)move the mutations beforehand
            raise ValueError("Cannot remove edges that have associated mutations")
    for remove_edge in edge_id_remove_list:
        e = ts.edge(remove_edge)
        edges_to_remove_by_child[e.child].append(e)

    # sort left-to-right for each child
    for k, v in edges_to_remove_by_child.items():
        edges_to_remove_by_child[k] = sorted(v, key=lambda e: e.left)
        # check no overlaps
        for e1, e2 in zip(edges_to_remove_by_child[k], edges_to_remove_by_child[k][1:]):
            assert e1.right <= e2.left

    # Sanity check: this means the topmost node will deal with modified edges left at the end
    assert ts.edge(-1).parent not in edges_to_remove_by_child
    
    new_edges = collections.defaultdict(list)
    tables = ts.dump_tables()
    tables.edges.clear()
    samples = set(ts.samples())
    # Edges are sorted by parent time, youngest first, so we can iterate over
    # nodes-as-parents visiting children before parents by using itertools.groupby
    for parent_id, ts_edges in itertools.groupby(ts.edges(), lambda e: e.parent):
        # Iterate through the ts edges *plus* the polytomy edges we created in previous steps.
        # This allows us to re-edit polytomy edges when the edges_to_remove are stacked
        edges = list(ts_edges)
        if parent_id in new_edges:
             edges += new_edges.pop(parent_id)
        if parent_id in edges_to_remove_by_child:
            for e in edges:
                assert parent_id == e.parent
                l = -1
                if e.id in edge_id_remove_list:
                    continue
                # NB: we go left to right along the target edges, reducing edge e as required
                for target_edge in edges_to_remove_by_child[parent_id]:
                    # As we go along the target_edges, gradually split e into chunks.
                    # If edge e is in the target_edge region, change the edge parent
                    assert target_edge.left > l
                    l = target_edge.left
                    if e.left >= target_edge.right:
                        # This target edge is entirely to the LHS of edge e, with no overlap
                        continue
                    elif e.right <= target_edge.left:
                        # This target edge is entirely to the RHS of edge e with no overlap.
                        # Since target edges are sorted by left coord, all other target edges
                        # are to RHS too, and we are finished dealing with edge e
                        tables.edges.append(e)
                        e = None
                        break
                    else:
                        # Edge e must overlap with current target edge somehow
                        if e.left < target_edge.left:
                            # Edge had region to LHS of target
                            # Add the left hand section (change the edge right coord)
                            tables.edges.add_row(left=e.left, right=target_edge.left, parent=e.parent, child=e.child)
                            e = e.replace(left=target_edge.left)
                        if e.right > target_edge.right:
                            # Edge continues after RHS of target
                            assert e.left < target_edge.right
                            new_edges[target_edge.parent].append(
                                e.replace(right=target_edge.right, parent=target_edge.parent)
                            )
                            e = e.replace(left=target_edge.right)
                        else:
                            # No more of edge e to RHS
                            assert e.left < e.right
                            new_edges[target_edge.parent].append(e.replace(parent=target_edge.parent))
                            e = None
                            break
                if e is not None:
                    # Need to add any remaining regions of edge back in 
                    tables.edges.append(e)
        else:
            # NB: sanity check at top means that the oldest node will have no edges above,
            # so the last iteration should hit this branch
            for e in edges:
                if e.id not in edge_id_remove_list:
                    tables.edges.append(e)
    assert len(new_edges) == 0
    tables.sort()
    return tables.tree_sequence()

def unsupported_edges(ts, per_interval=False):
    """
    Return the internal edges that are unsupported by a mutation.
    If ``per_interval`` is True, each interval needs to be supported,
    otherwise, a mutation on an edge (even if there are multiple intervals
    per edge) will result in all intervals on that edge being treated
    as supported.
    """
    edges_to_remove = np.ones(ts.num_edges, dtype="bool")
    edges_to_remove[[m.edge for m in ts.mutations()]] = False
    # We don't remove edges above samples
    edges_to_remove[np.isin(ts.edges_child, ts.samples())] = False

    if per_interval:
        return np.where(edges_to_remove)[0]
    else:
        keep = (edges_to_remove == False)
        for p, c in zip(ts.edges_parent[keep], ts.edges_child[keep]):
            edges_to_remove[np.logical_and(ts.edges_parent == p, ts.edges_child == c)] = False
        return np.where(edges_to_remove)[0]


# --- simulate data, introduce polytomies, date, generate figure

if __name__ == "__main__":

    overwrite_cache = False
    num_samples = 20000
    seed = 871234

    # simulate the binary trees
    cache = "../data/polytomy_bias_correction_binary.tsz"
    if not os.path.exists(cache) or overwrite_cache:
        ts = msprime.sim_ancestry(
            samples=num_samples,
            sequence_length=1e8,
            recombination_rate=1e-8,
            population_size=num_samples * 2,
            model=[msprime.DiscreteTimeWrightFisher(duration=100), msprime.StandardCoalescent()],
            random_seed=seed,
        )
        ts = msprime.sim_mutations(ts, rate=1.29e-8, random_seed=seed+1000)
        tszip.compress(ts, cache)
    else:
        ts = tszip.decompress(cache)

    # collapse unsupported edges into polytomies
    cache = "../data/polytomy_bias_correction_polytomy.tsz"
    if not os.path.exists(cache) or overwrite_cache:
        poly_ts = remove_edges(ts, unsupported_edges(ts)).simplify()
        tszip.compress(poly_ts, cache)
    else:
        poly_ts = tszip.decompress(cache)

    # date and write out node ages
    cache = "../data/polytomy_bias_correction.pkl"
    if not os.path.exists(cache) or overwrite_cache:
        import tsdate
        # settings are chosen to (1) be quick, 
        # (2) sufficiently rescale ages in recent time
        poly_ts_path = tsdate.date(poly_ts, mutation_rate=1.29e-8, max_iterations=2, rescaling_iterations=1, rescaling_intervals=20000)
        poly_ts_segs = tsdate.date(poly_ts, mutation_rate=1.29e-8, match_segregating_sites=True, max_iterations=2, rescaling_iterations=1, rescaling_intervals=20000)
        ts_path = tsdate.date(ts, mutation_rate=1.29e-8, max_iterations=2, rescaling_iterations=1)
        ts_segs = tsdate.date(ts, mutation_rate=1.29e-8, match_segregating_sites=True, max_iterations=2, rescaling_iterations=1)
        dates = {
            "pol_true": poly_ts.nodes_time.copy(),
            "pol_path": poly_ts_path.nodes_time.copy(),
            "pol_segs": poly_ts_segs.nodes_time.copy(),
            "bin_true": ts.nodes_time.copy(),
            "bin_path": ts_path.nodes_time.copy(),
            "bin_segs": ts_segs.nodes_time.copy(),
        } 
        pickle.dump(dates, open(cache, "wb"))
    else:
        dates = pickle.load(open(cache, "rb"))


    # plot true vs inferred node times under rescaling schemes, binary vs polytomy
    rows, cols = 2, 2
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols*3, rows*3), 
        constrained_layout=True, sharex=True, sharey=True,
    )
    axs[0, 0].hexbin(
        dates["bin_true"][ts.num_samples:], dates["bin_segs"][ts.num_samples:], 
        xscale="log", yscale="log", mincnt=1,
    )
    axs[0, 0].axline((1, 1), (1.1, 1.1), color="red", linestyle="dashed")
    axs[0, 0].text(
        0.01, 0.99, "binary, area rescaling",
        transform=axs[0, 0].transAxes, ha="left", va="top",
    )
    axs[0, 1].hexbin(
        dates["bin_true"][ts.num_samples:], dates["bin_path"][ts.num_samples:], 
        xscale="log", yscale="log", mincnt=1,
    )
    axs[0, 1].axline((1, 1), (1.1, 1.1), color="red", linestyle="dashed")
    axs[0, 1].text(
        0.01, 0.99, "binary, path rescaling",
        transform=axs[0, 1].transAxes, ha="left", va="top",
    )
    axs[1, 0].hexbin(
        dates["pol_true"][ts.num_samples:], dates["pol_segs"][ts.num_samples:], 
        xscale="log", yscale="log", mincnt=1,
    )
    axs[1, 0].axline((1, 1), (1.1, 1.1), color="red", linestyle="dashed")
    axs[1, 0].text(
        0.01, 0.99, "polytomies, area rescaling",
        transform=axs[1, 0].transAxes, ha="left", va="top",
    )
    axs[1, 1].hexbin(
        dates["pol_true"][ts.num_samples:], dates["pol_path"][ts.num_samples:], 
        xscale="log", yscale="log", mincnt=1,
    )
    axs[1, 1].axline((1, 1), (1.1, 1.1), color="red", linestyle="dashed")
    axs[1, 1].text(
        0.01, 0.99, "polytomies, path rescaling",
        transform=axs[1, 1].transAxes, ha="left", va="top",
    )
    fig.supxlabel("True node age")
    fig.supylabel("Estimated node age")
    plt.savefig("../figures/polytomy_bias_correction.pdf")




