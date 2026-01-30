import tsdate
import zipfile
import os
import xarray as xr
import pandas as pd
import numpy as np
import tskit
import shutil
from datetime import datetime
import sgkit as sg
import re
import tszip


def make_path(str):
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "tgp", str
    )


def save_csv(df, fn, float_format="%.7g"):
    start = datetime.now()
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "data",
        fn + ".csv.zst",
    )
    df.to_csv(
        path,
        sep="\t",
        compression={"method": "zstd", "level": 22, "threads": -1},  # max compression
        index=False,
        float_format=float_format,  # save space by using 7 d.p.
    )
    print(f"Saved data to {fn} in {datetime.now() - start} seconds")
    return path


def extract_metadata(
    input_path=make_path("20130606_g1k_3202_samples_ped_population.txt"),
    output_path=make_path("1kgp_metadata.csv"),
):
    """
    Extract metadata from the 1kgp sample file and save it as a CSV.
    """
    df = pd.read_csv(input_path, sep=" ")
    rename_dict = {
        "FamilyID": "family_id",
        "SampleID": "sample_id",
        "FatherID": "father_id",
        "MotherID": "mother_id",
        "Sex": "sex",
        "Population": "population",
        "Superpopulation": "superpopulation",
    }
    df.rename(columns=rename_dict, inplace=True)
    df.set_index("sample_id", drop=False, inplace=True)
    df["participant_type"] = "proband"
    df.loc[np.isin(df.sample_id, df.father_id), "participant_type"] = "father"
    df.loc[np.isin(df.sample_id, df.mother_id), "participant_type"] = "mother"
    df.loc[:, "father_id"] = df["father_id"].replace("0", "none")
    df.loc[:, "mother_id"] = df["mother_id"].replace("0", "none")
    df["sex"] = df["sex"].astype("str")
    df["sex"] = df["sex"].replace({"1": "male", "2": "female"})
    df.to_csv(output_path, index=False)


def get_proportional_sample(df, sample_size):
    """
    Take a balanced random sample from the 1kgp sample IDs, such that
    the proportion of samples in each superpopulation (AFR, EUR, etc.)
    is preserved as closely as possible
    """
    proportion = df["superpopulation"].value_counts(normalize=True)
    samples_per_group = (proportion * sample_size).round().astype(int)

    while samples_per_group.sum() != sample_size:
        diff = sample_size - samples_per_group.sum()
        if diff > 0:
            # Add the difference to the largest group
            samples_per_group[samples_per_group.index[0]] += diff
        else:
            max_group = samples_per_group.idxmax()
            samples_per_group[max_group] += (
                diff if samples_per_group[max_group] + diff > 0 else -diff
            )

    sampled_df = (
        df.groupby("superpopulation")
        .apply(
            lambda x: x.sample(n=samples_per_group[x.name], replace=False),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    proportion_related = (
        1 - sampled_df.participant_type.value_counts(normalize=True)["proband"]
    )
    print(f"Proportion of relatives for size {sample_size}: {proportion_related:.2f}")
    return sampled_df["sample_id"]


def generate_sample_lists(
    input_path=make_path("1kgp_metadata.csv"), seed=2, sample_sizes=[100, 300, 1500]
):
    np.random.seed(seed)
    df = pd.read_csv(input_path)
    sampled_ids = {size: get_proportional_sample(df, size) for size in sample_sizes}
    for size in sample_sizes:
        sampled_ids[size].to_csv(
            make_path(f"subsets/sample_lists/bal{size}_subset.csv"),
            header=None,
            index=None,
        )


def make_singletons_ds(zarr_path, subset_name, output_folder):
    """
    Given a zarr_vcf from the tsinfer-snakemake pipeline and a subset_name used
    for inference, this will determine all the SNP singletons in the subset (that
    are not singletons in the full dataset) and will store a bunch of useful
    information about them in an xarray dataset, which will be compressed
    into an output zipped zarr.
    """
    ds = sg.load_dataset(zarr_path, consolidated=False)
    singleton_mask_name = "variant_{subset_name}_subset_singleton_mask"

    singleton_mask = ds.data_vars[f"variant_{subset_name}_subset_derived_count"] != 1
    allele_lengths = ds.data_vars["variant_allele"].str.len()
    indel_mask = (allele_lengths != 1).any(dim="alleles")
    combined_mask = singleton_mask | indel_mask 
    ds.update({singleton_mask_name: combined_mask})

    genotype_ploidy = ds["call_genotype"].where(
        ~ds[singleton_mask_name].compute(), drop=True
    )
    neg_sample_mask = (~ds[f"sample_{subset_name}_subset_mask"]).compute()
    neg_sample_mask_broadcast = neg_sample_mask.broadcast_like(genotype_ploidy)
    genotype_subset_ploidy = genotype_ploidy.where(
        neg_sample_mask_broadcast, drop=False
    )
    
    genotype_subset_mask = genotype_subset_ploidy.sum(dim="ploidy") != 1
    sample_indexes = (~genotype_subset_mask).argmax(dim="samples")
    all_sample_ids = ds.data_vars["sample_id"]
    singleton_sample_ids = all_sample_ids.isel(samples=sample_indexes)

    genotype_subset_ploidy = genotype_ploidy.where(neg_sample_mask_broadcast, drop=True)
    genotype_array = genotype_subset_ploidy.stack(sample_nodes=["samples", "ploidy"])
    genotype_array = genotype_array.reset_index("sample_nodes")
    genotype_array = genotype_array.drop_vars(["samples", "ploidy"])

    phase = genotype_subset_ploidy.max(dim="samples").argmax(dim="ploidy")
    position = ds["variant_position"].where(
        ~ds[singleton_mask_name].compute(), drop=True
    )
    allele = ds["variant_allele"].where(~ds[singleton_mask_name].compute(), drop=True)

    ds_out = xr.Dataset()
    ds_out["genotype_array"] = genotype_array.astype("int8")
    ds_out["allele"] = allele.astype("object")
    ds_out["sample_id"] = singleton_sample_ids.astype("object")
    ds_out["phase"] = phase.astype("int8")
    ds_out["position"] = position.astype("float64")

    chunk_size = {"variants": 10000, "sample_nodes": -1}
    ds_out = ds_out.chunk(chunk_size)
    output_path = f"{output_folder}/{subset_name}_singletons.zarr"
    ds_out.to_zarr(output_path, mode="w")
    shutil.make_archive(output_path, "zip", output_path)
    print(f"Written dataset to {output_path}.zip")


def add_phased_singletons(
    ts,
    position,
    individual,
    ancestral_state,
    derived_state,
    phase,
):
    """
    Amend a tree sequence with singletons of known phase.
    """
    tables = ts.dump_tables()
    sites_id = {p: i for i, p in enumerate(ts.sites_position)}
    for pos, ind, ref, alt, phase in zip(
        position, individual, ancestral_state, derived_state, phase
    ):
        if pos in sites_id:
            if ref != ts.site(sites_id[pos]).ancestral_state:
                raise ValueError(
                    f"Existing site at position {pos} has a different ancestral state"
                )
            muts = ts.site(sites_id[pos]).mutations
            set_time = len(muts) and np.isfinite(muts[0].time)
        else:
            sites_id[pos] = tables.sites.add_row(position=pos, ancestral_state=ref)
            set_time = False
        site = sites_id[pos]
        node = ts.individual(ind).nodes[phase]
        time = ts.nodes_time[node] if set_time else tskit.UNKNOWN_TIME
        tables.mutations.add_row(
            site=site,
            node=node,
            time=time,
            derived_state=alt,
        )
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def make_singleton_ts(
    subset,
    inf_params,
    zip_folder=make_path("subsets/singletons"),
    trees_folder=make_path("subsets/trees"),
):
    """
    Takes a tree sequence inferred from a subset of the 1kgp data and adds singletons
    to it using the true phase or random phase. Produces three tree sequences:
     - singletons_phased = True with random phase
     - singletons_phased = True with true phase
     - singletons_phased = False with random phase
    """
    zip_path = os.path.join(zip_folder, f"{subset}_singletons.zarr.zip")
    output_path = os.path.splitext(zip_path)[0]
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
        ds = xr.open_dataset(output_path, engine="zarr")

    ts = tskit.load(os.path.join(trees_folder, f"{subset}-{inf_params}.trees"))

    # Keep just one allele for multi-allelic singletons
    dupe_pos = ds["position"].values
    duplicate_mask = ~pd.Series(dupe_pos).duplicated(keep="first").values
    assert np.sum(duplicate_mask) == len(np.unique(dupe_pos))
    left = min(ts.sites_position)
    right = max(ts.sites_position)
    oob_mask = (ds.position.values >= left) & (ds.position.values <= right)
    duplicate_mask_xarray = xr.DataArray(
        duplicate_mask, dims=["variants"], name="duplicate_position_mask"
    )
    oob_mask_xarray = xr.DataArray(oob_mask, dims=["variants"], name="oob_mask")
    ds.update(
        {"duplicate_position_mask": duplicate_mask_xarray, "oob_mask": oob_mask_xarray}
    )
    # construct arrays for adding singletons
    dedupe_ds = ds.where(ds.duplicate_position_mask & ds.oob_mask, drop=True)
    stn_pos = dedupe_ds["position"].values
    stn_sample_id = dedupe_ds["sample_id"].values
    stn_ancestral = dedupe_ds["allele"][:, 0].values
    stn_derived = dedupe_ds["allele"][:, 1].values
    stn_phase = dedupe_ds["phase"].values.astype(int)
    # map sample_ids to individual ids
    sample_id_dict = {}
    for individual in ts.individuals():
        sample_id = individual.metadata["variant_data_sample_id"]
        sample_id_dict[sample_id] = individual.id
    stn_individ = np.array([sample_id_dict[sample_id] for sample_id in stn_sample_id])

    true_phase_ts = add_phased_singletons(
        ts,
        position=stn_pos,
        individual=stn_individ,
        ancestral_state=stn_ancestral,
        derived_state=stn_derived,
        phase=stn_phase,
    )
    rand_phase_ts = tsdate.phasing.rephase_singletons(
        true_phase_ts, use_node_times=False, random_seed=1024
    )

    print(f"Dating singleton ARGs for subset {subset}")
    ts_dict = {}
    ts_dict["random_phase"] = tsdate.date(
        rand_phase_ts,
        mutation_rate=1.29e-08,
        method="variational_gamma",
        singletons_phased=True,
    )
    ts_dict["phase_agnostic"] = tsdate.date(
        rand_phase_ts,
        mutation_rate=1.29e-08,
        method="variational_gamma",
        singletons_phased=False,
    )
    ts_dict["true_phase"] = tsdate.date(
        true_phase_ts,
        mutation_rate=1.29e-08,
        method="variational_gamma",
        singletons_phased=True,
    )

    for name, ts in ts_dict.items():
       ts.dump(os.path.join(trees_folder, f'{subset}-{inf_params}-stn-{name}.trees'))

    return ts_dict


def compute_mutation_times(ts):
    tables = ts.dump_tables()
    tables.compute_mutation_times()
    return tables.tree_sequence()


def get_mutation_time(mut, what="posterior"):
    if what == "posterior":
        return mut.metadata["mn"]
    elif what == "midpoint":
        return mut.time
    else:
        raise ValueError("Invalid mutation time type")


def extract_mutation_times(
    full_ts,
    subset_ts,
    sites_pos,
    what="posterior",
    min_freq=None,
    max_freq=None,
):
    """
    Extracts mutation ages in TS with all samples and a subset of them at a given
    list of site positions. Based on the evaluation.mutation_time function in tsdate.
    """
    full_ts = compute_mutation_times(full_ts)
    subset_ts = compute_mutation_times(subset_ts)
    num_sites = len(sites_pos)
    full_sites_pos = full_ts.sites_position
    subset_sites_pos = subset_ts.sites_position
    assert np.all(np.isin(sites_pos, full_sites_pos))
    assert np.all(np.isin(sites_pos, subset_sites_pos))

    full_site_ids = np.where(np.isin(full_sites_pos, sites_pos))[0]
    subset_site_ids = np.where(np.isin(subset_sites_pos, sites_pos))[0]
    assert len(full_site_ids) == len(subset_site_ids)
    assert np.array_equal(full_site_ids, np.sort(full_site_ids))
    assert np.array_equal(subset_site_ids, np.sort(subset_site_ids))
    full_times = np.full(num_sites, np.nan)
    subset_times = np.full(num_sites, np.nan)

    for index, site_id in enumerate(full_site_ids):
        site = full_ts.site(site_id)
        muts = site.mutations
        if len(muts) == 1 and muts[0].edge != tskit.NULL:
            full_times[index] = get_mutation_time(muts[0], what=what)
    for index, site_id in enumerate(subset_site_ids):
        site = subset_ts.site(site_id)
        muts = site.mutations
        if len(muts) == 1 and muts[0].edge != tskit.NULL:
            subset_times[index] = get_mutation_time(muts[0], what=what)

    missing = np.logical_or(np.isnan(full_times), np.isnan(subset_times))
    full_times = full_times[~missing]
    subset_times = subset_times[~missing]

    if min_freq is not None or max_freq is not None:
        subset_site_ids = subset_site_ids[~missing]
        site_freq = np.zeros(subset_ts.num_sites)
        for tree in subset_ts.trees():
            for site in tree.sites():
                muts = site.mutations
                if len(muts) == 1:
                    mut = muts[0]
                    num_samples = tree.num_samples(mut.node)
                    site_freq[site.id] = num_samples
        freq = site_freq[subset_site_ids]
        is_freq = np.logical_and(freq >= min_freq, freq <= max_freq)
        full_times = full_times[is_freq]
        subset_times = subset_times[is_freq]

    return full_times, subset_times


def make_singleton_dataframe(ts_dict, subset, inf_params, what="midpoint"):
    """
    Creates the dataframe comparing three different phasing methods using mutation ages
    in the full TS as ground truth.
    """
    full_ts = tskit.load(
        make_path(f"subsets/trees/1kgp_all-{inf_params}-dated-1.29e-08.trees")
    )

    records = []
    shared_sites_pos = full_ts.sites_position
    for name, ts in ts_dict.items():
        shared_sites_pos = np.intersect1d(shared_sites_pos, ts.sites_position)

    for name, ts in ts_dict.items():
        full_times, subset_times = extract_mutation_times(
            full_ts=full_ts,
            subset_ts=ts,
            sites_pos=shared_sites_pos,
            what=what,
            min_freq=1,
            max_freq=1,
        )
        for t in subset_times:
            records.append({"subset": subset, "type": name, "time": t})
    for t in full_times:
        records.append({"subset": subset, "type": "full_ts", "time": t})

    return pd.DataFrame.from_records(records)


def make_tgp_singleton_data(
    *_ignored_pos,
    subsets=["1kgp_100", "1kgp_300", "1kgp_1500"],
    inf_params="chr20p-filterNton23-truncate-0-0-0-mm0-post-processed-simplified-SDN",
    time_type="midpoint",
    fn="tgp_singleton_data+0",
    **_ignored_kw,
):
    """
    Makes a long dataframe with mutation ages by subset and phasing method. The +0
    is added to support the plot.py naming convention. Called by make_figure_data.py.
    """
    dfs = []
    for subset in subsets:
        ts_dict = make_singleton_ts(subset, inf_params=inf_params)
        df = make_singleton_dataframe(
            ts_dict=ts_dict,
            subset=subset,
            inf_params=inf_params,
            what=time_type,
        )
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    save_csv(combined_df, fn=fn)


def make_tgp_subset_data(
    *_ignored_pos,
    subsets=["1kgp_100", "1kgp_300", "1kgp_1500"],
    inf_params="chr20p-filterNton23-truncate-0-0-0-mm0-post-processed-simplified-SDN",
    time_type="midpoint",
    fn="tgp_subset_data+0",
    **_ignored_kw,
):
    """
    Finds the shared sites across all provided subsets and stores the estimated ages of
    each in all subsets provided. Called by make_figure_data.py.
    """
    print(f"Loading tree sequences")
    full_ts = tskit.load(
        make_path(f"subsets/trees/1kgp_all-{inf_params}-dated-1.29e-08.trees")
    )
    shared_sites_pos = full_ts.sites_position
    ts_dict = {}
    for subset in subsets:
        ts = tskit.load(
            make_path(f"subsets/trees/{subset}-{inf_params}-dated-1.29e-08.trees")
        )
        ts = compute_mutation_times(ts)
        shared_sites_pos = np.intersect1d(shared_sites_pos, ts.sites_position)
        ts_dict[subset] = ts

    records = []
    for subset, ts in ts_dict.items():
        print(f"Extracting mutations for subset {subset}")
        full_times, subset_times = extract_mutation_times(
            full_ts=full_ts,
            subset_ts=ts,
            sites_pos=shared_sites_pos,
            what=time_type,
        )
        for t in subset_times:
            records.append({"subset": subset, "time": t})
    for t in full_times:
        records.append({"subset": "1kgp_all", "time": t})
    df = pd.DataFrame.from_records(records)
    save_csv(df, fn=fn)


def make_tgp_table_data(*_ignored_pos,
                        ts_folder_path="data/tgp/with_singletons",
                        csv_path="data/tgp/inference_stats.csv",
                        **_ignored_kw):
    """
    Pull summary statistics from all the trees inferred from 1000 Genomes data.
    """
    pattern = re.compile(r"all-([^-]+)-")
    records = []
    for fname in os.listdir(ts_folder_path):
        if fname.endswith(".trees"):
            ts = tskit.load(os.path.join(ts_folder_path, fname))
            match = pattern.search(fname)
            if match:
                chunk_name = match.group(1)
                zip_bytes = os.path.getsize(os.path.join(ts_folder_path, fname))
                first_pos = min(ts.sites_position)
                last_pos = max(ts.sites_position)
                records.append(
                    {
                        "region_name": chunk_name,
                        "first_site_position": first_pos,
                        "last_site_position": last_pos,
                        "sequence_length": last_pos - first_pos,
                        "number_of_sites": ts.num_sites,
                        "number_of_trees": ts.num_trees,
                        "number_of_nodes": ts.num_nodes,
                        "number_of_edges": ts.num_edges,
                        "number_of_mutations": ts.num_mutations,
                        "tszip_total_size_bytes": zip_bytes,
                    }
                )
    df = pd.DataFrame.from_records(records)
    numeric_cols = [
        "sequence_length",
        "number_of_sites",
        "number_of_trees",
        "number_of_nodes",
        "number_of_edges",
        "number_of_mutations",
        "tszip_total_size_bytes",
    ]
    total_row = (
        df[numeric_cols].sum()
        .to_dict()
    )
    total_row.update({
        "region_name": "Total",
        "first_site_position": np.nan,
        "last_site_position": np.nan,
    })
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")


