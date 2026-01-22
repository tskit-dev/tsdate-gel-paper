import tsdate
import os
import pandas as pd
import numpy as np
import tskit
import tszip
from liftover import get_lifter
import zipfile
import tempfile
import tarfile
import pyreadr
from tsdate.rescaling import count_mutations
from tsdate.phasing import mutation_frequency
from tqdm.auto import tqdm
import re
import json
from numba import njit
from pyfaidx import Fasta


@njit
def match_ancestral_allele(positions, refs, ancestral_array):
    """
    Fast function for checking ancestral alleles. We need to offset
    positions because of 1- vs 0-based indexing.
    """
    match_pos = []
    for pos, ref in zip(positions, refs):
        ancestral = ancestral_array[pos - 1]
        if ancestral == ref:
            match_pos.append(pos)
    return match_pos


def process_singletons(input, output, ancestral_array_path):
    """
    Processes the results of a bcftools script to extract singletons from the
    unphased VCFs. Excludes the following:
     - Indels
     - Variants with 1/1 genotypes
     - Variants where the REF allele is a singleton (0/1)
     - Variants where the REF allele doesn't match the ancestral allele
    """
    df = pd.read_csv(input, sep="\t", header=None)
    df.columns = ["sample_id", "chr", "position", "ref", "alt", "filter", "genotype"]
    df = df[(df.ref.str.len() == 1) & (df.alt.str.len() == 1)]
    homozyg_pos = df[df.genotype == "1/1"].position
    singletons = df[(~df.position.isin(homozyg_pos)) & (df.genotype != "1/1")]

    # Ancestral allele checking
    fasta = Fasta(ancestral_array_path)
    seq_name = list(fasta.keys())[0]
    ancestral_array = np.asarray(fasta[seq_name], dtype="U1")
    refs = singletons.ref.to_numpy().astype("U1")
    positions = singletons.position.to_numpy().astype("int32")
    match_pos = match_ancestral_allele(positions, refs, ancestral_array)
    matches_df = df[df.position.isin(match_pos)].copy()
    matches_df.reset_index(inplace=True, drop=True)
    matches_df.drop(["filter", "genotype"], inplace=True, axis=1)
    matches_df.to_csv(output, index=False)


def add_singletons(ts, input_csv):
    """
    Adds the singletons to the tree sequence and outputs the modified tree sequence.
    """
    df = pd.read_csv(input_csv)
    df.drop_duplicates(subset=["position"], keep="first", inplace=True)
    left = min(ts.sites_position)
    right = max(ts.sites_position)
    df = df[(df.position >= left) & (df.position <= right)]

    stn_pos = df["position"].values
    stn_sample_id = df["sample_id"].values
    stn_ancestral = df["ref"].values
    stn_derived = df["alt"].values

    # Map sample_ids to individual ids in ts
    meta = ts.individual(0).metadata
    if "variant_data_sample_id" in meta:
        id_field = "variant_data_sample_id"
    elif "sgkit_sample_id" in meta:
        id_field = "sgkit_sample_id"
    else:
        raise ValueError("Sample ID field not recognised")
        
    sample_id_dict = {}
    for individual in ts.individuals():
        sample_id = individual.metadata[id_field]
        sample_id_dict[sample_id] = individual.id

    stn_individ = np.array([sample_id_dict[sample_id] for sample_id in stn_sample_id])

    singletons_ts = tsdate.phasing.insert_unphased_singletons(
        ts,
        position=stn_pos,
        individual=stn_individ,
        ancestral_state=stn_ancestral,
        derived_state=stn_derived,
    )
    return singletons_ts


def batch_add_singletons(input_dir, input_csv, output):
    output_dir = output.parent
    files = [f for f in os.listdir(input_dir) if f.endswith(".trees")]
    num_chunks = len(files)
    assert num_chunks > 0

    for f in sorted(files):
        ts = tskit.load(os.path.join(input_dir, f))
        name, _ = os.path.splitext(f)
        stn_ts = add_singletons(ts=ts, input_csv=input_csv)
        stn_ts.dump(os.path.join(output_dir, f"{name}-stn.trees"))
    output.touch()


def run_tsdate(input, metadata_path, mutation_rate=1.29e-08):
    """
    Run both inside-outside and expectation propagation methods.
    """
    output_dir = metadata_path.parent
    input_dir = input.parent
    files = [f for f in os.listdir(input_dir) if f.endswith(".trees")]
    num_chunks = len(files)
    assert num_chunks > 0
    ins_out_dir = os.path.join(output_dir, "ins_out")
    ep_dir = os.path.join(output_dir, "ep")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ins_out_dir, exist_ok=True)
    os.makedirs(ep_dir, exist_ok=True)

    for i, f in enumerate(sorted(files)):
        ts = tskit.load(os.path.join(input_dir, f))
        pop_size = ts.diversity() / (4 * mutation_rate)
        ins_out_ts = tsdate.date(
            ts,
            mutation_rate=mutation_rate,
            method="inside_outside",
            population_size=pop_size,
            progress=True,
        )
        ins_out_path = os.path.join(ins_out_dir, f"ins_out_{i}.trees")
        ins_out_ts.dump(ins_out_path)

        ep_ts = tsdate.date(
            ts, mutation_rate=mutation_rate, method="variational_gamma", progress=True
        )
        ep_path = os.path.join(ep_dir, f"ep_{i}.trees")
        ep_ts.dump(ep_path)

    with open(metadata_path, "w") as meta_file:
        json.dump({"num_chunks": num_chunks}, meta_file)


def lift_over(converter, chrom, pos):
    converted = converter.convert_coordinate(chrom, pos)
    out = np.nan
    if (converted is not None) and len(converted) == 1:
        chrom_out, pos_out, _ = converted[0]
        if chrom == chrom_out:
            out = pos_out
    return out


def make_variant_column(df, ref_seq="hg38"):
    chrom = df["chrom"]
    pos = df[f"pos_{ref_seq}"].astype(int).astype(str)
    anc = df["ancestral"]
    der = df["derived"]
    df["variant"] = chrom + ":" + pos + anc + ">" + der
    df["variant"] = df["variant"].str.replace(" ", "", regex=False)
    return df


def make_ts_dataframe(ts, chr, ref_seq="hg38"):
    """
    Extract mutation age and frequency data from a tree sequence.
    """
    chrom = f"chr{chr}"
    af = mutation_frequency(ts) / ts.num_samples
    _, mut_edge = count_mutations(ts)
    mut_parent = ts.edges_parent[mut_edge]
    mut_child = ts.edges_child[mut_edge]
    upper_age = ts.nodes_time[mut_parent]
    lower_age = ts.nodes_time[mut_child]
    assert np.all(upper_age > lower_age)
    midpoint_age = (upper_age + lower_age) / 2

    records = []
    for i, mut in enumerate(ts.mutations()):
        site_id = mut.site
        site = ts.site(site_id)
        derived_state = mut.derived_state
        ancestral_state = site.ancestral_state
        if derived_state == "1" and ancestral_state == "0":
            derived_state = np.nan
            ancestral_state = np.nan
        pos = site.position
        variant = f"{chrom}:{int(pos)}{ancestral_state}>{derived_state}"
        records.append(
            {
                "chrom": chrom,
                f"pos_{ref_seq}": pos,
                "ancestral": ancestral_state,
                "derived": derived_state,
                "lower_age": lower_age[i],
                "upper_age": upper_age[i],
                "midpoint_age": midpoint_age[i],
                "derived_af": af[i],
                "variant": variant,
            }
        )
    df = pd.DataFrame.from_records(records)
    return df


def extract_tsdate_data(input, output, chr):
    with open(input, "r") as meta_file:
        metadata = json.load(meta_file)
    input_dir = input.parent
    num_chunks = metadata["num_chunks"]
    dfs = []
    for i in range(num_chunks):
        for method in ["ins_out", "ep"]:
            path = os.path.join(input_dir, method, f"{method}_{i}.trees")
            ts = tskit.load(path)
            df = make_ts_dataframe(ts, chr=chr)
            df["method"] = "tsdate_" + method
            dfs.append(df)
    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    full_df.to_csv(output, index=False)


def extract_geva_data(input, output):
    """
    Clean up the GEVA data by removing trailing spaces in column names
    and renaming them.
    """
    geva_df = pd.read_csv(input, compression="gzip", comment="#")
    geva_df.columns = geva_df.columns.str.strip()
    df = geva_df[
        [
            "Position",
            "AlleleRef",
            "AlleleAlt",
            "AgeMean_Mut",
            "AgeCI95Lower_Mut",
            "AgeCI95Upper_Mut",
        ]
    ].copy()
    df["chrom"] = "chr" + geva_df["Chromosome"].astype(str)
    df.rename(
        columns={
            "Position": "pos_hg19",
            "AlleleRef": "ancestral",
            "AlleleAlt": "derived",
            "AgeMean_Mut": "midpoint_age",
            "AgeCI95Lower_Mut": "lower_age",
            "AgeCI95Upper_Mut": "upper_age",
        },
        inplace=True,
    )
    converter = get_lifter("hg19", "hg38", cache="data", one_based=True)
    coords_in = df[["chrom", "pos_hg19"]].values
    pos_out = np.full(len(coords_in), np.nan, dtype=np.float64)
    for i, (chrom, pos) in enumerate(coords_in):
        pos_out[i] = lift_over(converter, chrom, pos)
    df["pos_hg38"] = pos_out
    df["method"] = "GEVA"
    df.dropna(inplace=True)
    df = make_variant_column(df)
    df.drop(columns=["pos_hg19"], inplace=True)
    df.to_csv(output, index=False)
    return df


def extract_singer_data(input, output, chr):
    ts = tszip.decompress(input)
    df = make_ts_dataframe(ts=ts, chr=chr)
    df["method"] = "singer"
    df.to_csv(output, index=False)


def extract_relate_data(
    input, chr, cols=["chrom", "BP", "lower_age", "upper_age", "ancestral/derived"]
):
    full_df = next(iter(input.values()))
    full_df["chrom"] = "chr" + full_df["CHR"].astype(str)
    cols = ["chrom", "BP", "lower_age", "upper_age", "ancestral/derived"]
    df = full_df.loc[full_df["chrom"] == f"chr{chr}", cols].copy()
    df.rename(
        columns={
            "BP": "pos_hg19",
            "lower_age": "pop_lower_age",
            "upper_age": "pop_upper_age",
        },
        inplace=True,
    )
    ad = (
        df["ancestral/derived"].astype(str).str.strip().str.split("/", n=1, expand=True)
    )
    df["ancestral"] = ad[0].str.strip()
    df["derived"] = ad[1].str.strip()
    converter = get_lifter("hg19", "hg38", cache="data", one_based=True)
    coords_in = df[["chrom", "pos_hg19"]].values
    pos_out = np.full(len(coords_in), np.nan, dtype=np.float64)
    for i, (chrom, pos) in enumerate(coords_in):
        pos_out[i] = lift_over(converter, chrom, int(pos))
    df["pos_hg38"] = pos_out
    df.drop(columns=["ancestral/derived"], inplace=True)
    df["pop_midpoint_age"] = (df["pop_lower_age"] + df["pop_upper_age"]) / 2
    df.dropna(inplace=True)
    df = make_variant_column(df, ref_seq="hg38")
    return df


def combine_relate_data(input, output, chr):
    """
    Iterate through the Relate zip files, extracting per-population mutation
    ages.
    """
    zip_paths = []
    for f in os.listdir(input):
        if re.match(r"allele_ages_[A-Z]+\.zip$", f):
            zip_paths.append(os.path.join(input, f))

    tasks = []
    for path in zip_paths:
        superpop_m = re.search(r"allele_ages_([A-Z]+)\.zip$", os.path.basename(path))
        if not superpop_m:
            raise ValueError(f"Could not find {path}.")
        superpop = superpop_m.group(1)
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                pm = re.search(r"allele_ages_([A-Z]+)\.RData$", name)
                if pm:
                    pop = pm.group(1)
                    tasks.append((path, name, superpop, pop))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    wrote_header = os.path.exists(output) and os.path.getsize(output) > 0

    for path, member, superpop, pop in tqdm(
        tasks, desc="Processing Relate RData files", unit="file"
    ):
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(path, "r") as z:
                z.extract(member, d)
            rpath = os.path.join(d, member)
            path = pyreadr.read_r(rpath)
            df = extract_relate_data(path, chr=chr)
            df["superpopulation"] = superpop
            df["population"] = pop
            df.to_csv(output, mode="a", header=not wrote_header, index=False)
            wrote_header = True


def aggregate_relate_data(input, output, metadata_path):
    """
    Combine Relate dataframes to obtain estimates of mutation ages using
    per population estimates weighted by population size.
    """
    df = pd.read_csv(input)
    meta_df = pd.read_csv(metadata_path)
    pop_weights = meta_df.groupby("population").size().to_dict()

    df["weight"] = df["population"].map(pop_weights)
    df["weighted_midpoint_age"] = df["pop_midpoint_age"] * df["weight"]

    agg_df = df.groupby("pos_hg38", observed=True, sort=False).agg(
        midpoint_age=("weighted_midpoint_age", "sum"),
        weight_sum=("weight", "sum"),
        lower_age=("pop_lower_age", "min"),
        upper_age=("pop_upper_age", "max"),
    )
    agg_df["midpoint_age"] = agg_df["midpoint_age"] / agg_df["weight_sum"]
    agg_df = agg_df.drop(columns="weight_sum").reset_index()

    base_df = df.drop(
        columns=[
            "pop_midpoint_age",
            "pop_lower_age",
            "pop_upper_age",
            "population",
            "superpopulation",
            "weight",
            "weighted_midpoint_age",
            "pos_hg19",
        ]
    ).drop_duplicates(subset="pos_hg38", keep="first")

    merged_df = base_df.merge(agg_df, on="pos_hg38", how="inner")
    merged_df["method"] = "relate"
    merged_df.to_csv(output, index=False)


def extract_coalNN_data(input, chr):
    with tarfile.open(input, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.isfile()), None)
        if member is None:
            raise ValueError("No file found inside coalNN archive")
        f = tar.extractfile(member)
        df = pd.read_csv(f, sep="\t")
    df.dropna(subset=["chromosome"], inplace=True)
    df["chrom"] = "chr" + df["chromosome"].astype(int).astype(str)
    df = df[df.chrom == f"chr{chr}"]
    df["ancestral_state"] = df["ancestral_state"].str.upper().replace([".", "-"], pd.NA)
    df = df[df["end(bp)"] - df["start(bp)"] == 1.0]
    df.rename(
        columns={
            "start(bp)": "pos_hg38",
            "age_estimate": "superpop_midpoint_age",
            "lower_age": "superpop_lower_age",
            "upper_age": "superpop_upper_age",
            "alternate_state": "derived",
            "ancestral_state": "ancestral",
        },
        inplace=True,
    )
    df = df[
        [
            "chrom",
            "pos_hg38",
            "ancestral",
            "derived",
            "superpop_lower_age",
            "superpop_upper_age",
            "superpop_midpoint_age",
        ]
    ]

    df = make_variant_column(df)
    return df


def combine_coalNN_data(input, output, chr):
    tar_paths = []
    for name in os.listdir(input):
        if re.match(r"[A-Z]+\.coalNN_age_annotations\.txt\.tar\.gz$", name):
            tar_paths.append(os.path.join(input, name))
    if not tar_paths:
        raise FileNotFoundError("No coalNN tar archives found")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    wrote_header = os.path.exists(output) and os.path.getsize(output) > 0

    for path in tqdm(sorted(tar_paths), desc="Processing coalNN tar", unit="file"):
        match = re.match(
            r"([A-Z]+)\.coalNN_age_annotations\.txt\.tar\.gz$", os.path.basename(path)
        )
        if not match:
            continue
        superpop = match.group(1)
        with tempfile.TemporaryDirectory() as d:
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(d)
            df = extract_coalNN_data(path, chr)
            df["superpopulation"] = superpop
            df.to_csv(output, mode="a", header=not wrote_header, index=False)
            wrote_header = True


def aggregate_coalNN_data(input, output, metadata_path):
    df = pd.read_csv(input)
    if df.empty:
        raise ValueError("No coalNN data to aggregate")
    meta_df = pd.read_csv(metadata_path)
    superpop_weights = meta_df.groupby("superpopulation").size().to_dict()
    df["weight"] = df["superpopulation"].map(superpop_weights)
    df["weighted_midpoint_age"] = df["superpop_midpoint_age"] * df["weight"]
    agg_df = df.groupby("pos_hg38", observed=True, sort=False).agg(
        midpoint_age=("weighted_midpoint_age", "sum"),
        weight_sum=("weight", "sum"),
        lower_age=("superpop_lower_age", "min"),
        upper_age=("superpop_upper_age", "max"),
    )
    agg_df["midpoint_age"] = agg_df["midpoint_age"] / agg_df["weight_sum"]
    agg_df = agg_df.drop(columns=["weight_sum"]).reset_index()
    base_df = df.drop(
        columns=[
            "superpop_midpoint_age",
            "superpop_lower_age",
            "superpop_upper_age",
            "superpopulation",
            "weight",
            "weighted_midpoint_age",
        ]
    ).drop_duplicates(subset="pos_hg38", keep="first")
    merged_df = base_df.merge(agg_df, on="pos_hg38", how="inner")
    merged_df["method"] = "coalNN"
    merged_df.to_csv(output, index=False)


def aggregate_data(input, output, methods, chr):
    dataframe_dir = input.parent
    dataframes = []
    for method in methods:
        path = os.path.join(dataframe_dir, f"{method}_chr{chr}.csv")
        df = pd.read_csv(path)
        assert len(df) > 0
        dataframes.append(df)
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
    merged_df.to_csv(output, index=False)
