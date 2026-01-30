import tempfile
import zipfile
from pathlib import Path
import os
import click
import numpy as np
import pandas as pd
import pyreadr
from liftover import get_lifter

from steps import lift_over


def _load_chunk_bounds(inference_stats_path):
    stats = pd.read_csv(inference_stats_path)
    stats = stats[stats["region_name"] != "Total"]
    stats["chr"] = stats["region_name"].str.extract(r"(\d+)").astype(int)
    bounds = {}
    for chr, group in stats.groupby("chr"):
        starts = group["first_site_position"].astype(float).to_numpy()
        ends = group["last_site_position"].astype(float).to_numpy()
        bounds[chr] = list(zip(starts, ends))
    return bounds


def _check_variant_chunk(df, chr, bounds):
    chunk_bounds = bounds.get(chr, [])
    in_chunk = np.zeros(len(df), dtype=bool)
    if chunk_bounds:
        pos = df["pos_hg38"].to_numpy()
        for start, end in chunk_bounds:
            in_chunk |= (pos >= start) & (pos <= end)
    return in_chunk


def extract_geva_age_by_chunk(chr, geva_path, output_path, bounds):
    """Extract GEVA midpoint ages labelled by chunk membership."""
    geva_file = Path(geva_path) / f"atlas.chr{chr}.csv.gz"
    geva_df = pd.read_csv(geva_file, compression="gzip", comment="#")
    geva_df.columns = geva_df.columns.str.strip()
    df = geva_df[["Position", "AgeMean_Mut"]].copy()
    df["chr"] = chr
    df.rename(
        columns={"Position": "pos_hg19", "AgeMean_Mut": "midpoint_age"}, inplace=True
    )

    converter = get_lifter("hg19", "hg38", cache="data", one_based=True)
    coords_in = df[["chr", "pos_hg19"]].values
    pos_out = np.full(len(coords_in), np.nan, dtype=np.float64)
    for i, (chr, pos) in enumerate(coords_in):
        chr_str = f"chr{chr}"
        pos_out[i] = lift_over(converter, chr_str, pos)
    df["pos_hg38"] = pos_out
    df.dropna(subset=["pos_hg38"], inplace=True)

    in_chunk = _check_variant_chunk(df, chr, bounds)
    print(f"Processed {chr} with {np.sum(in_chunk)} variants in tsinferred chunks")
    result = pd.DataFrame(
        {"age": df["midpoint_age"].to_numpy(), "chr": chr, "in_chunk": in_chunk}
    )
    header = not output_path.exists()
    result.to_csv(output_path, mode="a", header=header, index=False)


def build_geva_chunk_dataframe(
    geva_path,
    output_path,
    inference_stats_path="../data/inference-stats-and-timing.csv",
):
    """Build dataframe of GEVA midpoint ages with chunk labels."""
    bounds = _load_chunk_bounds(inference_stats_path)
    assert len(bounds) > 0
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()
    for chr in range(1, 23):
        extract_geva_age_by_chunk(chr, geva_path, output_path, bounds)


def _load_relate_population(relate_path, superpop, pop):
    zip_path = os.path.join(relate_path, f"allele_ages_{superpop}.zip")
    member = f"allele_ages_{pop}.RData"
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(zip_path, "r") as z:
            target = next(
                (m for m in z.namelist() if m.endswith(f"/{member}") or m == member),
                None,
            )
            if target is None:
                raise ValueError(f"{member} not found in {zip_path}")
            z.extract(target, d)
        rpath = Path(d) / target
        data = pyreadr.read_r(rpath)
    return next(iter(data.values()))


def extract_relate_age_by_chunk(df, converter, output_path, bounds):
    df = df.copy()
    df["chr"] = df["CHR"].astype(int)
    df["age"] = (df["lower_age"] + df["upper_age"]) / 2
    df.rename(columns={"BP": "pos_hg19"}, inplace=True)
    records = []
    for chr in range(1, 23):
        df_chr = df[df["chr"] == chr][["chr", "pos_hg19", "age"]].copy()
        if df_chr.empty:
            continue
        coords_in = df_chr[["chr", "pos_hg19"]].values
        pos_out = np.full(len(coords_in), np.nan, dtype=np.float64)
        for i, (chr, pos) in enumerate(coords_in):
            chr_str = f"chr{chr}"
            pos_out[i] = lift_over(converter, chr_str, int(pos))
        df_chr["pos_hg38"] = pos_out
        df_chr.dropna(subset=["pos_hg38"], inplace=True)
        in_chunk = _check_variant_chunk(df_chr, chr, bounds)
        records.append(
            pd.DataFrame(
                {
                    "chr": chr,
                    "pos_hg38": df_chr["pos_hg38"].to_numpy(),
                    "age": df_chr["age"].to_numpy(),
                    "in_chunk": in_chunk,
                }
            )
        )
        print(f"Processed {chr} with {len(df_chr)} variants")
        assert len(records) > 0
    result = pd.concat(records, ignore_index=True)
    header = not output_path.exists()
    result.to_csv(
        output_path,
        mode="a",
        compression={"method": "zstd", "level": 22, "threads": -1},
        header=header,
        index=False,
    )


def build_relate_chunk_dataframe(
    superpop,
    pop,
    relate_path,
    output_path,
    inference_stats_path="../data/inference-stats-and-timing.csv",
):
    bounds = _load_chunk_bounds(inference_stats_path)
    assert len(bounds) > 0
    df = _load_relate_population(relate_path=relate_path, superpop=superpop, pop=pop)
    converter = get_lifter("hg19", "hg38", cache="data", one_based=True)
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()
    extract_relate_age_by_chunk(df, converter, output_path, bounds)


@click.command()
@click.option("--method", type=click.Choice(["geva", "relate"]), required=True)
@click.argument("input_path")
@click.argument("output_path")
@click.option("--superpop", help="Superpopulation in 1kgp (e.g. AFR)")
@click.option("--pop", help="Population in 1kgp (e.g. YRI)")
@click.option(
    "--inference-stats",
    "inference_stats_path",
    default="../data/inference-stats-and-timing.csv",
    show_default=True,
)
def main(method, input_path, output_path, superpop, pop, inference_stats_path):
    if method == "geva":
        build_geva_chunk_dataframe(input_path, output_path, inference_stats_path)
    elif method == "relate":
        if superpop is None or pop is None:
            raise click.UsageError("Provide --superpop and --pop when method=relate")
        build_relate_chunk_dataframe(
            superpop, pop, input_path, output_path, inference_stats_path
        )


if __name__ == "__main__":
    main()
