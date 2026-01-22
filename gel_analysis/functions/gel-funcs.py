#!/usr/bin/env python3

import tskit
import tszip
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import numba
import zarr
import dataclasses
import os
#from cyvcf2 import VCF
#from pyliftover import LiftOver
import numpy as np
import pandas as pd

def no_edges(ts_path):
    ts = tszip.load(ts_path)
    n_edges = ts.num_edges
    return n_edges

def no_ancestors(ts_path):
    ts = tszip.load(ts_path)
    n_ancestors = ts.num_nodes - ts.num_samples
    return n_ancestors

def sample_from_posterior(muts, reps):
    """
    Sample mutation ages from the posterior distribution for a given set of mutation rows,
    ensuring that all sampled times are at least 1.
    """
    shape, scale = muts[["shape_time", "scale_time"]].values.T
    post_times = np.random.gamma(
        shape=np.tile(shape, (reps, 1)), scale=np.tile(scale, (reps, 1))
    )

    post_times_flat = np.sort(post_times.flatten())
    assert len(post_times_flat) == reps * len(muts)
    return post_times_flat

def sample_ages_by_category(df, variable, reps, num_bootstraps):
    """
    Bootstrap sample mutation ages from the posterior distribution for a given set of
    mutation rows, stratified by a categorical variable.
    """
    categories = df[variable].unique().tolist()
    times_dict = {}
    bootstrap_dict = {}

    for category in categories:
        muts = df[df[variable] == category]
        bootstrap_times = np.full((num_bootstraps, reps * len(muts)), np.nan)
        
        for i in range(num_bootstraps):
            muts_boot = muts.sample(n=len(muts), replace=True)
            bootstrap_times[i] = sample_from_posterior(muts_boot, reps)

        times = sample_from_posterior(muts, reps)
        times_dict[category] = times
        bootstrap_dict[category] = bootstrap_times

    return times_dict, bootstrap_dict

def qq_plot_data_from_df(df, variable, null_category, reps=100, num_bootstraps=100, num_points=100, conf_percent=95):
    """
    Generates a dataframe for plotting a Q-Q plot of mutation ages for each category
    against the null category, with bootstrapped confidence intervals.
    """

    # Generate times_dict and bootstrap_dict
    times_dict, bootstrap_dict = sample_ages_by_category(df, variable, reps, num_bootstraps)

    # Ensure the null category is processed first
    times_dict = dict(sorted(times_dict.items(), key=lambda item: (item[0] != null_category, item[0])))

    # Prepare null category times
    null_times = np.sort(times_dict[null_category])
    
    # Ensure all values > 1
    null_times = null_times[null_times > 0]  
    if len(null_times) == 0:
        raise ValueError("After filtering, no null times remain above 1. Consider adjusting the threshold.")

    # Collect plot data
    plot_data = []

    for category, times in times_dict.items():
        times = np.sort(times)
        
        bootstrap_times = bootstrap_dict[category]
        
        lower_quantiles = np.percentile(bootstrap_times, (100 - conf_percent) / 2, axis=0)
        upper_quantiles = np.percentile(bootstrap_times, 100 - (100 - conf_percent) / 2, axis=0)

        # Interpolate values for uniform sampling
        times_interp = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(times)), times)
        lower_interp = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(lower_quantiles)), lower_quantiles)
        upper_interp = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(upper_quantiles)), upper_quantiles)
        null_times_interp = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(null_times)), null_times)

        if category == null_category:
            for i in range(num_points):
                plot_data.append({
                    "null_times": null_times_interp[i],
                    "ratio": 1,
                    "lower": 1,
                    "upper": 1,
                    "category": category
                })
        else:
            for i in range(num_points):
                plot_data.append({
                    "null_times": null_times_interp[i],
                    "ratio": times_interp[i] / null_times_interp[i],
                    "lower": lower_interp[i] / null_times_interp[i],
                    "upper": upper_interp[i] / null_times_interp[i],
                    "category": category
                })

    return pd.DataFrame(plot_data)

def vcf_to_tsv(input_vcf, output_tsv, columns_to_extract=None, chain_file=None):
    """
    Extract specific columns from a VCF file and save as a TSV.
    Optionally lift over coordinates to a different genome build.

    Parameters:
        input_vcf (str): Path to the input VCF file.
        output_tsv (str): Path to the output TSV file.
        columns_to_extract (list): List of columns to extract (default: ['CHROM', 'POS', 'REF', 'ALT', 'QUAL']).
        chain_file (str): Path to the liftover chain file (optional). If provided, performs liftover.
    """
    # Default columns to extract if not provided
    if columns_to_extract is None:
        columns_to_extract = ['CHROM', 'POS', 'REF', 'ALT', 'QUAL']

    # Initialize liftover if chain_file is provided
    lo = LiftOver(chain_file) if chain_file else None

    # Data storage for TSV output
    data = []

    # Read the VCF file
    vcf = VCF(input_vcf)

    # Loop through each variant in the VCF
    for variant in vcf:
        chrom = variant.CHROM
        pos = variant.POS
        ref = variant.REF
        alt = ','.join(variant.ALT)  # Join multiple ALT alleles if present
        qual = variant.QUAL

        # Perform liftover if a chain file is provided
        if lo:
            lifted = lo.convert_coordinate(chrom, pos)
            if lifted:
                new_chrom, new_pos = lifted[0][0], lifted[0][1]
                chrom, pos = new_chrom, new_pos
            else:
                print(f"Liftover failed for {chrom}:{pos}")
                continue  # Skip this variant if liftover fails

        # Collect the data
        data.append([chrom, pos, ref, alt, qual])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns_to_extract)

    # Write to TSV
    df.to_csv(output_tsv, sep='\t', index=False)

    print(f"TSV file successfully written to {output_tsv}")

def adac_score(shape, scale, df, nsamples, direction="greater"):
    """
    Perform Monte Carlo sampling to calculate the Bayes Factor for a gamma distribution.

    Args:
        shape (float): Shape parameter for the focal allele.
        scale (float): Scale parameter for the focal allele.
        df (pd.DataFrame): DataFrame containing 'shape_time' and 'scale_time' columns for the dataset.
        nsamples (int): Number of Monte Carlo samples.
        direction (str): Direction of comparison ('greater' or 'lesser').

    Returns:
        float: Bayes Factor.
    """
    # Generate samples for the dataset
    theta_dat = np.array([
        np.random.gamma(row['shape_time'], row['scale_time'], nsamples) for _, row in df.iterrows()
    ]).flatten()
    
    # Generate samples for the mutation
    theta_mut = np.random.gamma(shape, scale, nsamples * len(df))
    
    # Calculate the fraction of samples meeting the condition
    if direction == "greater":
        theta_dif = np.mean(theta_mut > theta_dat)
    elif direction == "lesser":
        theta_dif = np.mean(theta_mut < theta_dat)
    else:
        raise ValueError("Direction must be 'greater' or 'lesser'")
    
    # Convert to Bayes Factor
    bf = theta_dif / (1 - theta_dif)
    
    # Handle special cases
    bf = min(nsamples, bf)  # Replace Inf with nsamples
    bf = max(0, bf)         # Replace -Inf with 0
    
    return bf

def ibd_between_carriers(ts_path, positions, min_span = 0):
    """
    Extracts IBD segments for mutation carriers at a specific position where the segments span the position.
    
    Parameters:
    - ts: Tree sequence object
    - position: Position of the mutation
    - min_span: Minimum span for IBD segments
    
    Returns:
    - A pandas DataFrame containing the filtered IBD segments
    """
    ts = tszip.load(ts_path)
    positions_set = set(positions)  # Use a set for O(1) lookups
    rows = []
    for v in tqdm(ts.variants(), total=ts.num_sites):
        v_position = int(v.position)
        if v_position not in positions_set:
            continue

        position_node = v.site.mutations[0].node
        genotypes = v.genotypes
        samples_index = np.where(genotypes == 1)[0]
        carriers = v.samples[samples_index].tolist()
        break

        carriers_ibd = ts.ibd_segments(within=carriers, store_pairs=True, store_segments=True, min_span=min_span)

        for pair, segments in carriers_ibd.items():
            for segment in segments:
                if segment.left <= v_position <= segment.right:
                    rows.append({
                        "sample1": int(pair[0]),
                        "sample2": int(pair[1]),
                        "left": segment.left,
                        "right": segment.right,
                        "node": segment.node,
                        "position_node": position_node,
                        "position": v_position,
                    })

    ibd_df = pd.DataFrame(rows)
    return(ibd_df)

def count_breakpoints_in_windows(ts_path, window_size=1000):
    """
    Count the number of breakpoints in 1kb windows along the genome.

    Args:
    - ts_path (str): Path to the tree sequence file.
    - window_size (int): The size of each window in bases (default is 1000).

    Returns:
    - pd.DataFrame: DataFrame with the count of breakpoints in each window.
    """
    # Load the tree sequence
    ts = tszip.load(ts_path)

    # Get the length of the sequence
    sequence_length = ts.sequence_length

    # Convert the breakpoints (map object) to a list or numpy array
    breakpoints = np.array(list(ts.breakpoints()))

    # Initialize lists to store window information and breakpoint counts
    windows = []
    breakpoints_counts = []

    # Iterate over the genome in 1kb windows with a tqdm progress bar
    for start in tqdm(range(0, int(sequence_length), window_size), 
                      total=int(sequence_length / window_size), 
                      desc="Processing windows", unit="window"):
        end = min(start + window_size, sequence_length)

        # Count the number of breakpoints in this window
        breakpoints_count = np.sum((breakpoints >= start) & (breakpoints < end))

        # Store the window start, end, and breakpoint count
        windows.append((start, end))
        breakpoints_counts.append(breakpoints_count)

    # Create a Pandas DataFrame to store the results
    df = pd.DataFrame({
        'window_start': [win[0] for win in windows],
        'window_end': [win[1] for win in windows],
        'breakpoint_count': breakpoints_counts
    })

    return df


def get_carriers_ids(ts_path, positions):
    # Load the tree sequence
    ts = tszip.load(ts_path)
    
     # Generate the sample_sets dictionary
    sample_sets = {
        ind.metadata.get("sgkit_sample_id") or ind.metadata.get("variant_data_sample_id"): ind.nodes 
        for ind in ts.individuals()
    }
    
    positions_set = set(positions)  # Use a set for O(1) lookups
    
    # Initialize a list to store the data
    rows = []
    
    # Iterate over the variants (sites)
    for v in tqdm(ts.variants(), total=ts.num_sites):
        v_position = int(v.position)
        
        # Check if the variant position is in the list of positions we're interested in
        if v_position not in positions_set:
            continue
        
        # Get the genotypes (assuming genotypes are in a format where 1 is the carrier)
        genotypes = v.genotypes
        
        # Find the indexes of the carriers (genotype == 1)
        samples_index = np.where(genotypes == 1)[0]
        
        # Get the carrier sample IDs based on the sample_sets dictionary
        for sample_idx in samples_index:
            carrier_sample = v.samples[sample_idx]  # Get the carrier sample ID
            
            # Now find the corresponding carrier_id from sample_sets
            carrier_id = None
            for sample, nodes in sample_sets.items():
                # Check if carrier_sample is in nodes (sample_sets stores arrays of nodes)
                if carrier_sample in nodes:
                    carrier_id = sample
                    break
            
            # Add data for each carrier at the current position, including sample_idx and carrier_id
            rows.append({
                "position": v_position,
                "carrier_sample_idx": carrier_sample,  # Store the actual sample index
                "carrier_id": carrier_id  # Get the carrier ID from sample_sets
            })
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(rows)
    
    return df

def mrca_span_carriers(ts_path, positions):
    ts = tszip.load(ts_path)
    positions_set = set(positions)
    rows = []
    for v in tqdm(ts.variants(), total=ts.num_sites):
        v_position = int(v.position)
        if v_position not in positions_set:
            continue
        position_node = v.site.mutations[0].node
        genotypes = v.genotypes
        samples_index = np.where(genotypes == 1)[0]
        carriers = v.samples[samples_index].tolist()
        
        if len(carriers) < 2:
            continue

        ibd_left = None
        ibd_right = None
        segments = []  # To store the spans

        # Iterate over trees in the tree sequence
        for tree in ts.trees():
            # Check if the MRCA of carriers matches the position_node
            if tree.mrca(*carriers) == position_node:
                if ibd_left is None:  # Start of a new segment
                    ibd_left = tree.interval[0]
                ibd_right = tree.interval[1]  # Update the right boundary
            else:
                if ibd_left is not None:  # End of a segment
                    segments.append((ibd_left, ibd_right))
                    ibd_left = None  # Reset for the next segment

        # Append the last segment if it extends to the end
        if ibd_left is not None:
            segments.append((ibd_left, ibd_right))
    
        for start, end in segments:
            rows.append({
                "position": v_position,
                "ibd_left": start,
                "ibd_right": end,
                "span": end - start,
                "num_carriers": len(carriers)
            })
    
    ibd_df = pd.DataFrame(rows)
    return(ibd_df)

def _merge_lr(lefts, rights):
    """Merge [left,right) intervals (numpy arrays) into a list of disjoint tuples."""
    if len(lefts) == 0:
        return []
    order = np.argsort(lefts)
    L = lefts[order].astype(float)
    R = rights[order].astype(float)
    out = []
    cur_l = L[0]; cur_r = R[0]
    for l, r in zip(L[1:], R[1:]):
        if l <= cur_r:  # overlap/adjacent
            if r > cur_r:
                cur_r = r
        else:
            out.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    out.append((cur_l, cur_r))
    return out

def _variant_carriers(ts, site_id):
    """
    Return (position_int, mutation_node, carriers_samples_ndarray)
    for a biallelic site; returns None if no derived carriers or <2 carriers.
    """
    # Preferred: construct Variant from a Site object
    try:
        site = ts.site(site_id)                   # Site object
        v = tskit.Variant(ts, site=site)          # <-- use 'site=', not 'site_id='
    except TypeError:
        # Ultra-conservative fallback: iterate until we hit site_id
        v = None
        for vv in ts.variants():
            if vv.site.id == site_id:
                v = vv
                break
        if v is None:
            return None

    # one mutation per site assumed; adapt if recurrent mutations occur
    if len(v.site.mutations) == 0:
        return None

    node = v.site.mutations[0].node
    g = v.genotypes
    idx = np.nonzero(g == 1)[0]       # carriers of ALT allele (code '1')
    if idx.size < 2:
        return None

    carriers = v.samples[idx]         # sample IDs corresponding to those genotypes
    pos_int = int(v.position)
    return (pos_int, node, carriers)

def mrca_span_carriers_fast(ts_path, positions):
    """
    For each requested variant position, find genomic segments where
    MRCA(carriers) == mutation node. Much faster than scanning all trees per variant:
      - restrict to intervals where the mutation node is a parent in edges
      - use tracked_samples + sample counts instead of tree.mrca(*carriers)
      - seek() to interval starts
    Returns a DataFrame with columns: position, ibd_left, ibd_right, span, num_carriers
    """
    ts = tszip.load(ts_path)

    # Map requested integer positions to site IDs (skip everything else)
    pos_set = set(int(p) for p in positions)
    site_ids = {}
    for s in ts.sites():
        ip = int(s.position)
        if ip in pos_set:
            site_ids[ip] = s.id
    if not site_ids:
        return pd.DataFrame(columns=["position", "ibd_left", "ibd_right", "span", "num_carriers"])

    # Convenience references to edge table columns as numpy arrays
    E = ts.tables.edges
    e_parent = np.asarray(E.parent, dtype=int)
    e_left   = np.asarray(E.left,   dtype=float)
    e_right  = np.asarray(E.right,  dtype=float)

    rows = []
    for pos in tqdm(sorted(site_ids.keys()), desc="variants", leave=False):
        site_id = site_ids[pos]
        info = _variant_carriers(ts, site_id)
        if info is None:
            continue
        v_position, node, carriers = info
        ncar = int(carriers.size)

        # Gather intervals where this node is an ancestor (parent in at least one edge)
        mask = (e_parent == node)
        if not np.any(mask):
            continue
        intervals = _merge_lr(e_left[mask], e_right[mask])

        # Iterate only over those intervals, with tracked_samples for O(1) checks
        tree_iter = ts.trees(tracked_samples=carriers)
        try:
            tree = tree_iter.__next__()  # robust under reticulate
        except AttributeError:
            tree = next(tree_iter)

        for left, right in intervals:
            tree.seek(left)
            ibd_left = None
            last_r = None

            while tree.interval[0] < right:
                # MRCA(carriers)==node  <=> subtree(node) == carriers
                if (tree.num_tracked_samples(node) == ncar) and (tree.num_samples(node) == ncar):
                    if ibd_left is None:
                        ibd_left = tree.interval[0]
                    last_r = tree.interval[1]
                else:
                    if ibd_left is not None:
                        rows.append({
                            "position": v_position,
                            "ibd_left": ibd_left,
                            "ibd_right": last_r,
                            "span": last_r - ibd_left,
                            "num_carriers": ncar
                        })
                        ibd_left = None

                # advance the Tree (not the iterator)
                if not tree.next():
                    break

            # close a trailing segment within this interval
            if ibd_left is not None:
                rows.append({
                    "position": v_position,
                    "ibd_left": ibd_left,
                    "ibd_right": last_r,
                    "span": last_r - ibd_left,
                    "num_carriers": ncar
                })

    return pd.DataFrame.from_records(rows)

def edge_span_mutations(ts_path, positions):
    """
    Calculate the edge span for mutations at given positions in a tree sequence.

    Args:
        ts_path (str): Path to the tree sequence file.
        positions (list): List of variant positions to process.

    Returns:
        pd.DataFrame: DataFrame with edge spans for the given positions.
    """
    ts = tszip.load(ts_path)
    positions_set = set(positions)  # Use a set for O(1) lookups
    rows = []
    positions_tree = ts.sites_position.astype(int)
    for tree in tqdm(ts.trees(), total = ts.num_trees):
        for mutation in tree.mutations():
            if positions_tree[mutation.site] not in positions_set:
                continue
            child_node = mutation.node
            child_node_md = ts.node(child_node).metadata
            m_parent = tree.parent(child_node)
            parent_node = ts.node(m_parent)
            parent_node_md = parent_node.metadata
            position = positions_tree[mutation.site]

            # Locate the edge associated with the mutation
            edge_index = mutation.edge  # This is valid if the mutation tracks the edge index
            if edge_index is not None:
                edge = ts.edge(edge_index)  # Retrieve the edge directly
                rows.append({
                    "position": position,
                    "edge_left": edge.left,
                    "edge_right": edge.right,
                    "child_node": child_node,
                    "child_node_mean_time": child_node_md.get("mn"),
                    "child_node_var_time": child_node_md.get("vr"),
                    "parent_node": parent_node.id,
                    "parent_node_mean_time": parent_node_md.get("mn"),
                    "parent_node_var_time": parent_node_md.get("vr")
                })
            else:
                rows.append({
                    "position": position,
                    "edge_left": None,
                    "edge_right": None,
                    "child_node": child_node,
                    "child_node_mean_time": child_node_md.get("mn"),
                    "child_node_var_time": child_node_md.get("vr"),
                    "parent_node": parent_node.id,
                    "parent_node_mean_time": parent_node_md.get("mn"),
                    "parent_node_var_time": parent_node_md.get("vr")
                })
        
    edge_df = pd.DataFrame(rows)
    return edge_df

def pop_pair_coal_rates(
    ts_path,
    pops_df,
    pop,
    time_max = 2e5,
    time_min = 1e1,
    bin_width = 10,
    ind_meta_strings = ["sgkit_sample_id", "variant_data_sample_id"]
):
    ts = tszip.load(ts_path)
    sample_sets = {
        ind.metadata.get("sgkit_sample_id") or ind.metadata.get("variant_data_sample_id"): ind.nodes 
        for ind in ts.individuals()
    }

    pop_inds = tuple(pops_df.loc[pops_df['pop'].isin(pop), 'sgkit_sample_id'])
    pop_sample_set = {k: sample_sets[k] for k in pop_inds}
    flattened_nodes = [int(node) for nodes in pop_sample_set.values() for node in nodes]
    
    time_windows = np.arange(time_min, time_max + bin_width, step=bin_width)
    time_windows = np.concatenate(([0], time_windows, [np.inf]))
    time_midpoints = (time_windows[:-1] + time_windows[1:]) / 2 # Midpoints for each window

    pair_coal = ts.pair_coalescence_rates(sample_sets = [flattened_nodes], time_windows = time_windows)
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame({
        'time_windows': time_midpoints,
        'pair_coal_rates': pair_coal
    })
    return results_df

def cross_pop_pair_coal_rates(
    ts_path,
    pops_df,
    pop1,
    pop2,
    time_max = 2e5,
    time_min = 1e1,
    bin_width = 10
):
    ts = tszip.load(ts_path)
    sample_sets = {
    ind.metadata.get("sgkit_sample_id") or ind.metadata.get("variant_data_sample_id"): ind.nodes 
    for ind in ts.individuals()
    }
    
    pop_inds1 = tuple(list(pops_df.loc[pops_df['pop'].isin(pop1), 'sgkit_sample_id']))
    pop_sample_set1 = {k: sample_sets[k] for k in pop_inds1}
    flattened_nodes_focal = np.array([int(node) for nodes in pop_sample_set1.values() for node in nodes], dtype=np.int32)
    
    pop_inds2 = tuple(list(pops_df.loc[pops_df['pop'].isin(pop2), 'sgkit_sample_id']))
    pop_sample_set2 = {k: sample_sets[k] for k in pop_inds2}
    flattened_nodes_comp = np.array([int(node) for nodes in pop_sample_set2.values() for node in nodes], dtype=np.int32)
    print()

    # Create time windows using bin_width
    time_windows = np.arange(time_min, time_max + bin_width, step=bin_width)
    time_windows = np.concatenate(([0], time_windows, [np.inf]))
    time_midpoints = (time_windows[:-1] + time_windows[1:]) / 2 # Midpoints for each window

    pair_coal = ts.pair_coalescence_rates(sample_sets = [flattened_nodes_focal, flattened_nodes_comp], time_windows = time_windows)
    results_df = pd.DataFrame({
        'time_windows': time_midpoints,
        'pair_coal_rates': pair_coal
    })
    return results_df

def indiv_vs_all_pair_coal_counts(
    ts_path,
    focal_sample_id,
    time_max=2e5,
    time_min=1e1,
    bin_width=10,
    id_fields=("sgkit_sample_id", "variant_data_sample_id")
):
    """
    Compute pair coalescence counts between ONE focal individual and ALL other individuals.

    Parameters
    ----------
    ts_path : str
        Path to a tszip-compressed tree sequence.
    focal_sample_id : str
        The focal individual's sample ID (matched against metadata keys in `id_fields`).
    time_max, time_min, bin_width : float
        Time windowing parameters (same units as ts time; typically generations).
    id_fields : tuple[str]
        Metadata keys to try (in order) for matching the provided sample ID.

    Returns
    -------
    pandas.DataFrame with columns: t_left, t_right, t_mid, pair_coal_rate
    """
    ts = tszip.load(ts_path)

    # Build {sample_id -> sample nodes[]} from individual metadata
    id_to_nodes = {}
    for ind in ts.individuals():
        meta = ind.metadata if isinstance(ind.metadata, dict) else {}
        sid = None
        for k in id_fields:
            if meta and meta.get(k):
                sid = meta[k]
                break
        if sid is None:
            continue
        # keep only nodes that are samples
        nodes = [n for n in ind.nodes if ts.node(n).is_sample()]
        if nodes:
            id_to_nodes[sid] = np.array(nodes, dtype=np.int32)

    if focal_sample_id not in id_to_nodes:
        raise KeyError(
            f"Focal ID '{focal_sample_id}' not found in individual metadata fields {id_fields}."
        )

    focal_nodes = id_to_nodes[focal_sample_id]
    all_sample_nodes = np.array(ts.samples(), dtype=np.int32)
    comp_nodes = all_sample_nodes[~np.isin(all_sample_nodes, focal_nodes)]
    if comp_nodes.size == 0:
        raise ValueError("No comparator nodes found (tree sequence may only contain the focal individual).")

    # Time windows (prepend 0 and append +inf to match your previous convention)
    edges = np.arange(time_min, time_max + bin_width, step=bin_width, dtype=float)
    time_windows = np.concatenate(([0.0], edges, [np.inf]))
    time_midpoints = (time_windows[:-1] + time_windows[1:]) / 2.0

    # Compute pair coalescence rates
    pair_coal = ts.pair_coalescence_counts(
        sample_sets=[focal_nodes, comp_nodes],
        time_windows=time_windows
    )

    if getattr(pair_coal, "ndim", 1) == 1:
        counts = pair_coal
    else:
        counts = pair_coal[0, 1, :]

    results_df = pd.DataFrame({
        "time_windows": time_midpoints,
        "pair_coal_counts": counts
    })
    return results_df

def dump_times_data_frame(ts_path, na="NA"):
    # load ts
    ts = tszip.load(ts_path)
    # Samples and positions quick
    n_samples = ts.num_samples
    positions = {'position': ts.sites_position.astype(int)}
    df = pd.DataFrame(positions)

    # To append
    derived_states = []
    ancestral_states = []
    derived_counts = []
    times = []
    mean_times = []
    var_times = []

    for v in tqdm(ts.variants(), total = ts.num_sites):
        s = v.site
        if len(s.mutations) != 1:
            print(
                f"Skipped site at position {s.position} due to",
                ("multiple mutations" if len(s.mutations) > 1 else "no mutations")
            )
            continue
        m = s.mutations[0]
        # No longer need to decode JSON
        md = m.metadata
        derived_states.append(m.derived_state)
        ancestral_states.append(s.ancestral_state)
        derived_counts.append(np.sum(v.genotypes))
        times.append(m.time)
        mean_times.append(md.get("mn", na))
        var_times.append(md.get("vr", na))
    
    df.insert(1, "ancestral_state", ancestral_states, True)
    df.insert(2, "derived_state", derived_states, True)
    df.insert(3, "AC_ts", derived_counts, True)
    df.insert(4, "AN_ts", n_samples, True)
    df.insert(5, "time", times, True)
    df.insert(6, "mean_time", mean_times, True)
    df.insert(7, "var_time", var_times, True)
    return(df)

def dump_pop_allele_counts_data_frame(ts_path, pops_path, colid = "ts"):
    ts = tszip.load(ts_path)
    sample_sets = {
        ind.metadata.get("sgkit_sample_id") or ind.metadata.get("variant_data_sample_id"): ind.nodes 
        for ind in ts.individuals()
    }
    pops_df = pd.read_csv(pops_path, sep = "\t")

    def get_allele_counts(ts, sa):
        allele_counts = []
        for v in tqdm(ts.variants(samples = sa), total = ts.num_sites):
            s = v.site
            if len(s.mutations) != 1:
                print(
                    f"Skipped site at position {s.position} due to",
                    ("multiple mutations" if len(s.mutations) > 1 else "no mutations")
                )
                continue
            allele_counts.append(np.sum(v.genotypes))
        return(allele_counts)

    # Get positions for joining with ts_df
    positions = {'position': ts.sites_position.astype(int)}
    df = pd.DataFrame(positions)

    # Add an "all" option
    populations = list(set(pops_df['pop']))
    for pop in populations:  
        pop_inds = tuple(list(pops_df.loc[pops_df['pop'] == pop]['sgkit_sample_id'])) 
        pop_sample_set = {k: sample_sets[k] for k in pop_inds}
        pop_samples_arr = np.concatenate(list(pop_sample_set.values()))
        print(f'Extracting allele counts for {len(pop_samples_arr)/2} individuals assigned to ' + pop)
        pop_allele_counts = get_allele_counts(ts = ts, sa = pop_samples_arr)
        df.insert(0, "AC_" + pop + "_" + colid, pop_allele_counts, True)
        df.insert(0, "AN_" + pop + "_" + colid, len(pop_samples_arr), True)
    return(df)

def parent_is_root(ts_path):
    ts = tszip.load(ts_path)
    sites_below_root = []
    positions = ts.sites_position.astype(int)
    for tree in tqdm(ts.trees(), total = ts.num_trees):
        for mutation in tree.mutations():
            if tree.parent(mutation.node) in tree.roots:
                sites_below_root.append(mutation.site)
    positions = [positions[i] for i in sites_below_root]
    return(positions)

def find_cpg_sites(fasta_file):
    fasta = Fasta(fasta_file)
    seq_name = list(fasta.keys())[0]
    sequence = np.asarray(fasta[seq_name], dtype="U1")
    positions = []
    for i in tqdm(range(len(sequence) - 1)):
        # Check if current and next characters form a CpG site
        if sequence[i] == 'C' and sequence[i + 1] == 'G':
            positions.append(i+1) #we need to 1-index
    return positions

def get_ind_hts(ts_path, ind_id):
    ts = tszip.load(ts_path)
    sample_sets = {
    ind.metadata.get("sgkit_sample_id") or ind.metadata.get("variant_data_sample_id"): ind.nodes 
    for ind in ts.individuals()
    }
    sa = sample_sets[ind_id]
    gt = ts.genotype_matrix(samples = sa)
    gt_df = pd.DataFrame(gt)
    positions = {'position': ts.sites_position.astype(int)}
    gt_df['position'] = pd.Series(positions['position'])
    return(gt_df)

def get_zarr_pos_ind_indices(zarr_path):
    # Open group (prefer consolidated if available)
    try:
        root = zarr.open_consolidated(zarr_path, mode="r")
    except Exception:
        root = zarr.open_group(zarr_path, mode="r")

    # Read arrays
    pos = root["variant_position"][:]
    samples = root["sample_id"][:]

    pos = np.asarray(pos, dtype=np.int64)
    if samples.dtype.kind in {"S", "O"}:
        try:
            samples = np.char.decode(samples, "utf-8")
        except Exception:
            samples = samples.astype(str)
    else:
        samples = samples.astype(str)
    samples = samples.tolist()

    # Build DataFrames with an explicit 'index' column
    pos_df = pd.DataFrame({
        "index": np.arange(pos.shape[0], dtype=np.int64),
        "values": pos
    })
    samples_df = pd.DataFrame({
        "index": np.arange(len(samples), dtype=np.int64),
        "values": samples
    })

    return [pos_df, samples_df]

@dataclasses.dataclass
class GenotypeCounts:
    hom_ref: list
    hom_alt: list
    het: list
    ref_count: list

@numba.njit(
    "void(int64, int8[:,:,:], b1[:], b1[:], int32[:], int32[:], int32[:], int32[:])"
)

def count_genotypes_chunk_subset(
    offset, G, variant_mask, sample_mask, hom_ref, hom_alt, het, ref_count
):
    # NB Assuming diploids and no missing data!
    index = offset
    for j in range(G.shape[0]):
        if variant_mask[j]:
            for k in range(G.shape[1]):
                if sample_mask[k]:
                    a = G[j, k, 0]
                    b = G[j, k, 1]
                    if a == b:
                        if a == 0:
                            hom_ref[index] += 1
                        else:
                            hom_alt[index] += 1
                    else:
                        het[index] += 1
                    ref_count[index] += (a == 0) + (b == 0)
            index += 1

def classify_genotypes_subset(
    call_genotype, variant_mask, sample_mask
):
    m = np.sum(variant_mask)

    # Use zarr arrays to get mask chunks aligned with the main data
    # for convenience.
    z_variant_mask = zarr.array(variant_mask, chunks=call_genotype.chunks[0])
    z_sample_mask = zarr.array(sample_mask, chunks=call_genotype.chunks[1])

    het = np.zeros(m, dtype=np.int32)
    hom_alt = np.zeros(m, dtype=np.int32)
    hom_ref = np.zeros(m, dtype=np.int32)
    ref_count = np.zeros(m, dtype=np.int32)
    j = 0
    # We should probably skip to the first non-zero chunk, but there probably
    # isn't much difference unless we have a huge number of chunks, and we're
    # only selecting a tiny subset
    for v_chunk in range(call_genotype.cdata_shape[0]):
        variant_mask_chunk = z_variant_mask.blocks[v_chunk]
        count = np.sum(variant_mask_chunk)
        if count > 0:
            for s_chunk in range(call_genotype.cdata_shape[1]):
                sample_mask_chunk = z_sample_mask.blocks[s_chunk]
                if np.sum(sample_mask_chunk) > 0:
                    G = call_genotype.blocks[v_chunk, s_chunk]
                    count_genotypes_chunk_subset(
                        j,
                        G,
                        variant_mask_chunk,
                        sample_mask_chunk,
                        hom_ref,
                        hom_alt,
                        het,
                        ref_count,
                    )
            j += count
    return GenotypeCounts(hom_ref, hom_alt, het, ref_count)

def check_individual_genotype(
    df, zarr_path, zarr_pos, zarr_ind
):
    root = zarr.open(zarr_path, mode="r")
    call_genotype = root['call_genotype']
    mask = df['position'].isin(zarr_pos)
    df = df[mask]
    ids = df.sgkit_sample_id.values
    pos_ind_map = pd.Series(df.position.values, index=df.sgkit_sample_id).to_list()
    sample_varies = np.ones(len(df), dtype=bool)
    ## Only want specific pos/ind genotypes in df, so first set mask all all false
    ## Check if zarr_pos_dict == call_genotype.shape[0]
    
    for ind, pos in tqdm(enumerate(pos_ind_map), total = len(pos_ind_map)):
        ind_index = zarr_ind.index(ids[ind])
        sample_mask = np.zeros(call_genotype.shape[1], dtype=bool)
        sample_mask[ind_index] = True
        pos_index = zarr_pos.index(pos)
        variant_mask = np.zeros(call_genotype.shape[0], dtype=bool)
        variant_mask[pos_index] = True
        gt = classify_genotypes_subset(call_genotype, variant_mask, sample_mask)
        if gt.hom_ref.all() == 0:
            sample_varies[ind] = False
    df.insert(0, "sample_varies", sample_varies, True)
    return(df)

def local_gnn_sample(ts_path, ind_id, pops_df):
    ts = tszip.load(ts_path)
    # pops_df = pd.read_csv(pops_path, sep = "\t")
    
    sample_sets = {
    ind.metadata.get("sgkit_sample_id") or ind.metadata.get("variant_data_sample_id"): ind.nodes 
    for ind in ts.individuals()
    }
    populations = list(set(pops_df['pop']))
    region_sample_set = []
    for pop in populations:  
        pop_inds = tuple(list(pops_df.loc[pops_df['pop'] == pop]['sgkit_sample_id'])) 
        pop_sample_set = {k: sample_sets[k] for k in pop_inds}
        pop_samples_list = list(np.concatenate(list(pop_sample_set.values())))
        region_sample_set.append(pop_samples_list)

    def local_gnn(ts, focal, reference_sets):
        reference_set_map = np.zeros(ts.num_nodes, dtype=int) - 1
        for k, reference_set in enumerate(reference_sets):
            for u in reference_set:
                if reference_set_map[u] != -1:
                    raise ValueError("Duplicate value in reference sets")
                reference_set_map[u] = k

        K = len(reference_sets)
        A = np.zeros((len(focal), ts.num_trees, K))
        lefts = np.zeros(ts.num_trees, dtype=float)
        rights = np.zeros(ts.num_trees, dtype=float)
        parent = np.zeros(ts.num_nodes, dtype=int) - 1
        sample_count = np.zeros((ts.num_nodes, K), dtype=int)

        # Set the intitial conditions.
        for j in range(K):
            sample_count[reference_sets[j], j] = 1

        num_edge_diffs = sum(1 for _ in ts.edge_diffs())
        for t, ((left, right),edges_out, edges_in) in tqdm(enumerate(ts.edge_diffs()), total = num_edge_diffs):
            for edge in edges_out:
                parent[edge.child] = -1
                v = edge.parent
                while v != -1:
                    sample_count[v] -= sample_count[edge.child]
                    v = parent[v]
            for edge in edges_in:
                parent[edge.child] = edge.parent
                v = edge.parent
                while v != -1:
                    sample_count[v] += sample_count[edge.child]
                    v = parent[v]

            # Process this tree.
            for j, u in enumerate(focal):
                focal_reference_set = reference_set_map[u]
                p = parent[u]
                lefts[t] = left
                rights[t] = right
                while p != tskit.NULL:
                    total = np.sum(sample_count[p])
                    if total > 1:
                        break
                    p = parent[p]
                if p != tskit.NULL:
                    scale = 1 / (total - int(focal_reference_set != -1))
                    for k, reference_set in enumerate(reference_sets):
                        n = sample_count[p, k] - int(focal_reference_set == k)
                        A[j, t, k] = n * scale
        return (A, lefts, rights)

    for ind in ts.individuals():
        md = ind.metadata
        if md.get('sgkit_sample_id') == ind_id or md.get('variant_data_sample_id') == ind_id:
            counter = 1
            df_list = []
            for j, node in enumerate(ind.nodes):
                print("Haplotype " + str(counter))
                A, left, right = local_gnn(ts, [node], region_sample_set)
                df = pd.DataFrame(data = A[0], columns = populations)
                df["left"] = left
                df["right"] = right
                # Remove rows with no difference in GNN to next row
                keep_rows = ~(df.iloc[:, 0:len(populations)].diff(axis=0) == 0).all(axis=1)
                df = df[keep_rows]
                df.insert(0, "haplotype", counter, True)
                df_list.append(df)
                counter += 1
            df_out = pd.concat(df_list)
    return(df_out)