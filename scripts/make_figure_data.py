import argparse
import collections
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from cyvcf2 import VCF
import numpy as np
import tszip
from tqdm.auto import tqdm
import pandas as pd
from scipy import stats
import make_tgp_data as mtd
import tskit
import myvariant
from liftover import get_lifter

data_dir = Path(__file__).resolve().parent.parent / "data"

def save_csv(df, fn, float_format='%.7g'):
    # standard way to save CSV dataframes
    start = datetime.now()
    path = data_dir / f"{fn}.csv.zst"
    df.to_csv(
        path,
        sep="\t",
        compression={'method': 'zstd', 'level': 22, 'threads': -1},  # max compression
        index=False,
        float_format=float_format,  # save space by using 7 d.p.
    )
    print(f"Saved data to {fn} in {datetime.now() - start} seconds")
    return path

Times = collections.namedtuple("Times", "true, midpoint")

def mut_times(ts, site_positions):
    """
    Return 2 numpy arrays of true and midpoint times
    """
    nodes_time = ts.nodes_time
    edges_parent = ts.edges_parent
    
    true = np.zeros(len(site_positions))
    midpoint = np.zeros(len(site_positions))
    assert np.all(np.diff(site_positions) > 0)
    site_stack = list(site_positions)
    for s in ts.sites():
        if s.position == site_stack[0]:
            i = len(site_positions) - len(site_stack)
            del site_stack[0]
            assert len(s.mutations) == 1
            mut = s.mutations[0]
            true[i] = mut.time
            midpoint[i] = (nodes_time[mut.node] + nodes_time[edges_parent[mut.edge]]) / 2
            if len(site_stack) == 0:
                break
    return Times(true=true, midpoint=midpoint)

def make_sampling_sim_data(prefix, num_samples=None):
    data = {}

    def midpoint_times(tree_seq):
        # return an array of mutation times that lie halfway along the edge
        times = np.zeros_like(tree_seq.mutations_time)
        node_times = tree_seq.nodes_time
        for tree in tqdm(tree_seq.trees(), desc="Mid mut times"):
            for m in tree.mutations():
                times[m.id] = node_times[[m.node, tree.parent(m.node)]].mean()
        return times

    files = collections.defaultdict(dict)
    for fn in os.listdir(data_dir):
        if (m := re.match(fr"sampling_sim_([a-z]+)\+(\d+).dated.tsz", fn)):
            sampling_scheme = m.group(1)
            if num_samples is None:
                n = int(m.group(2))
                files[n][sampling_scheme] = fn
            elif num_samples == int(m.group(2)):
                files[num_samples][sampling_scheme] = fn
    if len(files) == 1:
        n = list(files.keys())[0]
    elif len(files) == 0:
        raise ValueError("No valid files found")
    else:
        raise ValueError(
            f"Multiple files (n={list(files.keys())}, please specify which via --select"
        )
    sampling = {}
    for sim, fn in files[n].items():
        ts = tszip.load(data_dir / fn)
        sampling[sim] = {p.metadata["name"]: len(ts.samples(p.id)) for p in ts.populations()}
        sim_ts = tszip.load(data_dir / fn.replace(".dated.tsz", ".tsz"))
        assert np.all(ts.sites_position == sim_ts.sites_position)
        
        sim_times = midpoint_times(sim_ts)
        freq_count = np.zeros(ts.num_sites)
        nodes_population = sim_ts.nodes_population
        pop_map = {pop.metadata["name"]:pop.id for pop in sim_ts.populations()}

        freq, freq_count, true, infer, pop, sel_coef = [], [], [], [], [], []
        for v1, v2 in tqdm(
            zip(sim_ts.variants(), ts.variants()),
            total=ts.num_sites,
            desc="Counting frequencies",
        ):
            m1 = v1.site.mutations
            m2 = v2.site.mutations
            if len(m1) == 1 and len(m2) == 1:
                # Todo: we could allow multiple mutations in the inferred ts and take e.g. the most recent
                true.append(sim_times[m1[0].id])
                infer.append(m2[0].metadata["mn"])
                freq.append(v1.frequencies()[m1[0].derived_state])
                freq_count.append(v1.counts()[m1[0].derived_state])
                assert m1[0].derived_state == m2[0].derived_state
                pop.append(nodes_population[m1[0].node])
                try:
                    sel_coef.append(m1[0].metadata['mutation_list'][0]['selection_coeff'])
                except (KeyError, TypeError):
                    sel_coef.append(0)
        print(
            f"Used {len(true)} single-mutation sites ({len(true)/ts.num_sites:.2%}%)"
            f" for sim {sim}"
        )
        data[sim] = pd.DataFrame({
            "true_midtime": true,
            "inferred_time": infer,
            "freq": freq,
            "freq_count": freq_count,
            f"pop={json.dumps(pop_map, separators=(',', ':'))}": pop,
            "s": sel_coef,        
        })
    # stack the data frames, with an index column: "balanced" or "unbalanced"
    df = pd.concat(data.values(), keys=data.keys()).reset_index(level=1, drop=True)
    # use a single-letter per sampling scheme: check each starts with a unique letter
    assert len(set(k[0] for k in sampling.keys())) == len(sampling)
    colname = "sampling_scheme=" + json.dumps(sampling, separators=(',', ':'))
    df[colname] = ""
    for idx in df.index.unique():
        df.loc[idx, colname] = idx[0]
    
    # Save as a csv and reduce filesize
    save_csv(df, data_dir / f"{prefix}_data+{n}", float_format='%.5g')

def make_pedigree_sim_data(prefix, seed_used=None):
    files = {}
    base = "simulated_chrom_17"
    cutoffs = [np.inf, 1000]

    orig_ts = tszip.load(data_dir / f"{base}.ts.tsz")

    tsdate_version = None
    for fn in os.listdir(data_dir):
        # e.g. simulated_chrom_17-285+None_123.tsdate0.2.2inside_outside.tsz 
        if (m := re.match(fr"{base}-(\d+)\+(\d+|None)_(\d+).tsdate([\d\.]+)(\w*).tsz", fn)):
            num_samp, ep_iterations, seed, version, method= m.groups()
            num_samp = int(num_samp)
            ep_iterations = None if ep_iterations == "None" else int(ep_iterations)
            seed = int(seed)

            if seed_used is None or seed == seed_used:
                if tsdate_version is None:
                    tsdate_version = version
                elif tsdate_version != version:
                    raise ValueError(
                        f"Multiple files with seed {seed} using different tsdate versions {tsdate_version} and {version}."
                    )
                if seed not in files:
                    files[seed] = {}
                if method not in files[seed]:
                    files[seed][method] = {}
                if ep_iterations not in files[seed][method]:
                    files[seed][method][ep_iterations] = {}
                if num_samp in files[seed][method][ep_iterations]:
                    raise ValueError(
                        f"Multiple files with seed {seed}, ep_iterations {ep_iterations}, num_samp {num_samp} and method {method}"
                    )
                files[seed][method][ep_iterations][num_samp] = fn
    if len(files) == 1:
        seed = list(files.keys())[0]
        files = files[seed]
        methods = list(files.keys())
        print(f"Using seed {seed} with methods {methods}")
        # sort the files by ep_iterations (None first) and then by num_samples
        for method, datafiles in files.items():
            files[method] = {
                k: dict(sorted(v.items()))
                for k, v in sorted(datafiles.items(), key=lambda ep: np.inf if ep[0] is None else ep[0])
            }
    elif len(files) == 0:
        raise ValueError("No valid files found")
    else:
        raise ValueError(
            f"Multiple seeds (s={list(files.keys())}, please specify which via --select"
        )
    
    use_ep_its = list(files[methods[0]].keys())[-1]
    for method, datafiles in files.items():
        if use_ep_its not in datafiles:
            raise ValueError(
                f"ep_iterations ({use_ep_its}) not found for method {method}"
            )

    min_ep = None
    for meth, datafiles in files.items():
        for ep in datafiles.keys():
            if min_ep is None or (ep is not None and ep < min_ep):
                min_ep, samples_used, ep_meth = ep, list(datafiles[ep].keys()), meth
    if min_ep is None:
        raise ValueError(f"No valid files found with 1 iteration and seed {seed}")

    # Data for accuracy, as measured by the correlation coefficient
    # Use the one with the default ep_iterations
    data = dict(sample_size=[], cutoff=[],  method=[], corr_coef=[], spearmans=[], nmuts=[])
    num_samples_map = files[ep_meth][None]
    print(f"Accuracy in sims with default EP iterations: n={list(num_samples_map.keys())}")
    dated_times = Times(
        true=pd.DataFrame(columns=[f'{i}' for i in num_samples_map.keys()]),
        midpoint=pd.DataFrame(columns=[f'{i}' for i in num_samples_map.keys()])
    )
    positions = None
    for n, path in tqdm(num_samples_map.items(), desc="finding common positions"):
        ts = tszip.load(data_dir / path)
        assert ts.num_samples == n
        if positions is None:
            positions = np.zeros(ts.num_sites, dtype=bool)
            for site in ts.sites():
                if len(site.mutations) == 1:
                    orig_site = orig_ts.site(position=site.position)
                    if len(orig_site.mutations) == 1:
                        positions[site.id] = True
            positions = ts.sites_position[positions]
        assert np.all(np.isin(positions, ts.sites_position))
        dated_times.true[str(n)], dated_times.midpoint[str(n)] = mut_times(ts, positions)
    orig_times = mut_times(orig_ts, positions)

    full_df = pd.DataFrame()  # Full datasets for scatterplots & violin plots
    for src, name in enumerate(Times._fields):
        x = np.log10(orig_times[src])
        if name == "midpoint":
            full_df["true"] = x
        for c in cutoffs:
            use = orig_times.true < c
            if name == "midpoint" and np.isfinite(c):
                full_df[f"cutoff{c:g}"] = use
            for n in tqdm(num_samples_map.keys()):
                y = np.log10(dated_times[src][str(n)])
                data["sample_size"].append(n)
                data["cutoff"].append(c)
                data["method"].append(name)
                data["nmuts"].append(np.sum(use))
                data["corr_coef"].append(np.corrcoef(x[use], y[use])[0,1])
                data["spearmans"].append(stats.spearmanr(x[use], y[use]).statistic)
                if name == "midpoint" and not np.isfinite(c):
                    full_df[n] = y
    save_csv(pd.DataFrame.from_dict(data), f"{prefix}_ACCURACY_data+{seed}")
    save_csv(full_df, f"{prefix}_FULL_data+{seed}", float_format='%.5g')

def make_validation_aDNA_data(prefix, chromosome=None):
    import tsdate  # do this within the function as it is slow to load

    if chromosome is None:
        # If no chromosome is specified, use the one from the AADR data
        chromosome = 20

    anno_fn = data_dir / "adna" / "aadr_v62.0_1240K_public.anno"
    vcf_fn = data_dir / "adna" / f"aadr_v62.0_1240K_public_hg38_chr{chromosome}.bcf"
    if not anno_fn.exists() or not vcf_fn.exists():
        raise ValueError(
            f"Could not find AADR data files {anno_fn} or {vcf_fn}, please "
            f"run `make chr CHROM={chromosome}` in the data/adna directory first."
        )
    # Load the ARG data
    tsdata = {
        "vgamma": {"pos": [], 'lower': [], 'upper': [], "derived_state": []},
        "inout": {"pos": [], 'lower': [], 'upper': [], "derived_state": []},
    }
    tgp_dir = data_dir / "tgp"
    match = "all-chr([0-9]+)(.*)-filterNton23-truncate-0-0-0-mm0-post-processed-simplified-SDN-singletons-dated-metadata.trees.tsz"
    for fn in os.listdir(tgp_dir):
        m = re.search(str(match), fn)
        if not m or int(m.group(1)) != chromosome:
            continue
        base_ts = tszip.load(tgp_dir / fn)
        print("Using tsdated ages from", fn)
        prov = json.loads(base_ts.provenance(-1).record)
        if prov['software']['name'] != "tsdate":
            raise ValueError(
                "Expected the last provenance entry to be `tsdate`, "
                f"got {prov['software']['name']} in {fn}"
            )
        mu = prov['parameters']['mutation_rate']
        old_ts = tsdate.inside_outside(
            base_ts,
            mutation_rate=mu,
            population_size=base_ts.trim().diversity() / (4 * mu),
            ignore_oldest_root=True,
            progress=True
        )
        for ts, data in zip((base_ts, old_ts), (tsdata["vgamma"], tsdata["inout"])):
            nodes_time = ts.nodes_time
            edges_parent = ts.edges_parent
            single_mutation_sites = np.bincount(ts.mutations_site, minlength=ts.num_sites) == 1
            single_mut_site_muts = np.isin(ts.mutations_site, np.where(single_mutation_sites)[0])
            data["pos"] = np.concatenate((
                data["pos"], ts.sites_position[single_mutation_sites]
            ))
            data["lower"] = np.concatenate((
                data["lower"], nodes_time[ts.mutations_node[single_mut_site_muts]]
            ))
            data["upper"] = np.concatenate((
                data["upper"], nodes_time[edges_parent[ts.mutations_edge[single_mut_site_muts]]]
            ))
            ds = ts.mutations_derived_state[single_mut_site_muts]
            if len(data["derived_state"]) == 0:
                data["derived_state"] = ds
            else:
                data["derived_state"] = np.concatenate((data["derived_state"], ds))
            assert len(data["upper"]) == len(data["pos"]) == len(data["derived_state"])
    assert np.all(tsdata["vgamma"]["pos"] == tsdata["inout"]["pos"])
    assert np.all(tsdata["vgamma"]["derived_state"] == tsdata["inout"]["derived_state"])

    # Load the AADR data
    df = pd.read_csv(anno_fn, sep="\t", index_col=0, low_memory=False)
    vcf = VCF(vcf_fn)
    sample_names = [s.split("_", 1)[1] for s in vcf.samples]
    dates = df.loc[
        sample_names,
        'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]',
    ]


    oldest = {}  # Save a dict of {pos: {'A': age, 'C': age}, ...}
    n_variants = None
    try:
        n_variants = vcf.num_records
    except ValueError:
        pass
    for variant in tqdm(vcf, total=n_variants, desc="Finding oldest"):
        # only look at biallelic SNPs (these are all that is present in the AADR anyway)
        if len(variant.ALT) != 1:
            continue
        genos = np.array(variant.genotypes)
        genos_a = genos[:, 0]
        genos_b = genos[:, 1]
        # If there is a zero, it has the ref
        oldest_ref = np.max(dates[np.logical_or(genos_a == 0, genos_b == 0)])
        oldest_alt = np.max(dates[np.logical_or(genos_a == 1, genos_b == 1)])
        oldest[variant.POS] = {variant.REF: oldest_ref, variant.ALT[0]: oldest_alt}
    
    x = np.array([
        oldest.get(int(p), {}).get(ds, np.nan)
        for p, ds in zip(tsdata["vgamma"]["pos"], tsdata["vgamma"]["derived_state"])
    ])
    use = np.isfinite(x)

    data = {
        "position": tsdata["vgamma"]["pos"][use],
        "AADR": x[use],
        "tsdate_vgamma_lower": tsdata["vgamma"]["lower"][use],
        "tsdate_vgamma_upper": tsdata["vgamma"]["upper"][use],
        "tsdate_inout_lower": tsdata["inout"]["lower"][use],
        "tsdate_inout_upper": tsdata["inout"]["upper"][use],
    }
    save_csv(pd.DataFrame.from_dict(data), f"{prefix}_TSDATE_data+{chromosome}")
    


def make_validation_OOA_data(prefix, chromosome=None):
    # tree seqs used below obtained from running validate_real_tsdate.py
    n_timebins = 100
    max_time = 100_000  # generations ~= 3M years
    time_intervals = np.logspace(0, np.log10(max_time), n_timebins)
    time_intervals = np.concatenate(([0], time_intervals, [np.inf]))


    phlash_names = {"Africa": "AFR", "Europe-Middle East":"EUR",  "East Asia": "EAS"}

    # For the moment we don't run Phlash ourselves, but use the output from the paper
    raw_df = pd.read_csv("data/Phlash_fig7b.csv")
    data = {'pop': [], 'iicr': [], 'gens': []}
    for name in phlash_names.keys():
        df = raw_df[raw_df['pop'] == name]
        data['pop'] += [phlash_names[name]] * len(df)
        data['iicr'] += df['Ne'].tolist()
        data['gens'] += (df['years']).tolist()  # It appears as if this is mislabelled "years", but is actually generations
    phlash_data = save_csv(pd.DataFrame.from_dict(data), f"{prefix}_PHLASH_data")

    chromosomes = collections.defaultdict(list)
    for fn in os.listdir(f"{data_dir}/tgp/"):
        if fn.startswith("chr") and fn.endswith(".tsz"):
            chromosomes[int(re.match(r"chr(\d+)", fn).group(1))].append(fn)
    
    if chromosome is None:
        chrs = list(chromosomes.keys())
        if len(chrs) == 1:
            chromosome = chrs[0]
        else:
            raise ValueError( f"Multiple chromosomes found ({chrs}), specify one via --select")
    if chromosome not in chromosomes:
        raise ValueError(f"No files found for chromosome {chromosome}")
    # copy the same file over, so that plot.py knows that the same PHLASH data 
    # can be used for this chromosome
    fn_copy = str(phlash_data).replace("PHLASH_data", f"PHLASH_data+{chromosome}")
    if os.path.exists(fn_copy):
        os.remove(fn_copy)
    shutil.copy(phlash_data, fn_copy)
    total = {}
    total_length = {}
    for fn in chromosomes[chromosome]:
        ts = tszip.load(f"{data_dir}/tgp/{fn}").trim()
        min_pos, max_pos = 0, ts.sequence_length
        for tree in ts.trees():
            if tree.index == 0 and tree.num_edges == 0:
                min_pos = tree.interval[1]
            elif tree.index == ts.num_trees - 1 and tree.num_edges == 0:
                max_pos = tree.interval[0]
            else:
                assert tree.num_edges != 0  # Check no missing regions in the middle
        pops = collections.defaultdict(list)
        for ind in ts.individuals():
            pops[ind.metadata["superpopulation"]] += list(ind.nodes)
        for name, samples in pops.items():
            rates = ts.pair_coalescence_rates(
                sample_sets = [np.array(samples)],
                time_windows=time_intervals,
            )
            rates[np.isfinite(rates) == False] = 0
            L = max_pos - min_pos
            rates *= L
            if name not in total:
                total[name] = rates
                total_length[name] = L
            else:
                total[name] += rates
                total_length[name] += rates
        
    data = {'pop': [], 'iicr': [], 'gens': []}
    for name in phlash_names.values():
        total[name] /= total_length[name]
        for rate, lo, hi in zip(total[name], time_intervals[:-1], time_intervals[1:]):
            data['pop'].append(name)
            data['iicr'].append(1 / (2 * rate))
            data['gens'].append((lo + hi) / 2)
    save_csv(pd.DataFrame.from_dict(data), f"{prefix}_TSDATE_data+{chromosome}")

def _process_inversion_snps(df):
    # Remove SNP with unknown position
    df = df[df.snp_name != "E_TAUIVS11_10"].copy()
    mv = myvariant.MyVariantInfo()

    res = mv.querymany(
        df.snp_name.tolist(),
        scopes="dbsnp.rsid",
        fields="chrom,hg38.start",
        as_dataframe=True,
        assembly="hg38",
    )
    res.drop_duplicates(subset=['hg38.start'], inplace=True)
    pos_hg38 = res.loc[df.snp_name, 'hg38.start'].values.astype(np.float64)
    df["pos_hg38"] = pos_hg38
    df.set_index("pos_hg38", inplace=True, drop=False)
    return df

def _process_relate_data(df_in):
    def lift_over(converter, pos_in, chrom="chr17"):
        pos_out = np.full(len(pos_in), np.nan, dtype=np.float64)
        for i, pos in enumerate(pos_in):
            converted = converter.convert_coordinate(chrom, pos)
            if (converted is not None) and len(converted) == 1:
                _, out, _ = converted[0]
            assert ~np.isnan(out)
            pos_out[i] = out
        return pos_out
    
    df = df_in.copy()
    df.columns = df.columns.str.lower()
    df = df[df.in_span == 1]
    df.rename(columns={"start": "start_hg19", "end": "end_hg19"}, inplace=True)
    converter = get_lifter("hg19", "hg38", one_based=True)
    df["start_hg38"] = lift_over(converter, df.start_hg19)
    df["end_hg38"] = lift_over(converter, df.end_hg19)
    return df

def _process_inversion_ts(ts,
                          snp_df,
                          relate_df,
                          num_genome_windows = 200,
                          num_time_windows = 50):
    """
    Finds all the carriers of the H1 and H2 inversion haplotypes in an inferred
    ARG using marker SNP data from Donnelly et al (2010) in the dataframe snp_df, 
    computing pairwise coalescence rates between carriers and non-carriers.
    """
    snps = snp_df["pos_hg38"].values
    matching_sites = np.intersect1d(ts.sites_position, snps)
    snp_sites = np.where(np.isin(ts.sites_position, matching_sites))[0]

    variants = tskit.Variant(ts)
    sample_list = []
    skips = 0
    for site_id in snp_sites:
        variants.decode(site_id)
        site = variants.site
        row = snp_df.loc[site.position]
        ancestral = site.ancestral_state
        derived = site.mutations[0].derived_state
        h1 = row.h1_allele
        h2 = row.h2_allele
        if (ancestral == h1 and derived == h2):
            samples = np.where(variants.genotypes == 1)[0]
        elif (ancestral == h2 and derived == h1):
            samples = np.where(variants.genotypes == 0)[0]
        else:
            skips += 1
            continue
        sample_list.append(set(samples))
   
    print(f'{skips} of {len(snp_sites)} have at least one allele that is not H1/H2')
    all_sample_carriers = np.array(list(set.union(*sample_list)), dtype="int")
    all_samples = np.arange(ts.num_samples)
    non_carriers = np.setdiff1d(all_samples, all_sample_carriers)
    carriers = np.array(
        list(set.intersection(*sample_list)), dtype="int"
    ) 
    assert len(carriers) > 0
    max_time = relate_df["time_high"].max() + 1e5
    min_time = relate_df["time_low"].min()

    x_min = min(relate_df.start_hg38)
    x_max = max(relate_df.end_hg38)
    x_incr = (x_max - x_min) / (num_genome_windows - 1)
    assert x_incr > 0

    genome_windows = np.arange(x_min, x_max, x_incr)
    genome_windows = np.sort(np.append(genome_windows, [0, ts.sequence_length]))
    time_windows = np.logspace(
        np.log10(min_time), np.log10(max_time), num=num_time_windows - 1
    )
    time_windows = np.sort(np.append(time_windows, [0, np.inf]))
    rates = ts.pair_coalescence_rates(
        sample_sets=[carriers, non_carriers], windows=genome_windows, time_windows=time_windows
    )
    # Adjust time windows for plot
    time_windows = time_windows[::-1].copy()
    time_windows[0] = max_time
    matrix = np.log(np.transpose(rates[:, ::-1]) + 1e-15)
    min_not_nan = np.nanmin(matrix)
    coal_mat = np.where(np.isnan(matrix), min_not_nan, matrix)

    dict = {"coal_mat": coal_mat,
            "genome_windows": genome_windows,
            "time_windows": time_windows}
    return dict

def make_validation_inversion_data(prefix,
                                   select=None,
                                   ts_path="tgp/with_singletons/all-chr17q45M~83M-filterNton23-truncate-0-0-0-mm0-post-processed-simplified-SDN-singletons-dated-metadata.trees",
                                   snp_csv="chr17inversion/donnelly_et_al_table_2.csv",
                                   relate_csv="chr17inversion/chr17q21.31_time_plot.csv",
                                   chromosome=17,
                            ):
    """
    Requires the inferred tree sequence for the 45-83mb chunk of chr17 and
    internet access (to look up positions of SNPs by RSid).
    Outputs three numpy arrays needed to plot pairwise coalescence events
    (time_windows, genomic_windows and coal_mat) and a dataframe of Relate age 
    estimates for the inversion.
    """
    ts = tskit.load(f"{data_dir}/{ts_path}")
    snp_df_in = pd.read_csv(f"{data_dir}/{snp_csv}")
    relate_df_in = pd.read_csv(f"{data_dir}/{relate_csv}")

    snp_df = _process_inversion_snps(snp_df_in)
    relate_df = _process_relate_data(relate_df_in)
    array_dict =  _process_inversion_ts(ts=ts, snp_df=snp_df, relate_df=relate_df)

    for desc, arr in array_dict.items():
        fn = f'{prefix}_{desc}_data+{chromosome}.csv'
        np.savetxt(
            os.path.join(data_dir, fn), arr, delimiter=",", fmt="%.7g",
        )
    save_csv(df=relate_df, fn=f'{prefix}_relate_data+{chromosome}')

choices = {
    "pedigree_sim": make_pedigree_sim_data,
    "sampling_sim": make_sampling_sim_data,
    "validation_aDNA": make_validation_aDNA_data,
    "validation_OOA": make_validation_OOA_data,
    "validation_inversion": make_validation_inversion_data,
    "tgp_singleton": mtd.make_tgp_singleton_data,
    "tgp_subset": mtd.make_tgp_subset_data,
    "tgp_table": mtd.make_tgp_table_data,
}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Generate data for figures')
    argparser.add_argument(
        'figure',
        help='The figure to produce',
        choices=list(choices.keys()),
    )
    argparser.add_argument(
        '--select', '-s',
        type=int,
        help=(
            "Parameter used to narrow down which files to select. "
            "In the case of pedigree_sim, this is the random seed used. "
            "In the case of sampling_sim, this is the random seed used. If not "
            "specified, the script will check there is only one valid set of files."
        ),
        default=None,  # default involves checking if there are any existing files
    )


    args = argparser.parse_args()
    choices[args.figure](args.figure, args.select)
 