import pandas as pd
import numpy as np


def format_time(seconds):
    """Format time in seconds to appropriate units (s, m, h, d, y)"""
    if pd.isna(seconds):
        return "N/A"

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:  # Less than 1 hour
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    elif seconds < 86400:  # Less than 1 day
        hours = seconds / 3600
        return f"{hours:.2f}h"
    elif seconds < 31536000:  # Less than 1 year
        days = seconds / 86400
        return f"{days:.2f}d"
    else:
        years = seconds / 31536000
        return f"{years:.2f}y"


def format_number(num):
    """Format large numbers with commas"""
    if pd.isna(num):
        return "N/A"

    return f"{int(num):,}"


def format_bytes(bytes_val):
    """Format bytes with appropriate suffixes (B, KB, MB, GB, TB)"""
    if pd.isna(bytes_val):
        return "N/A"

    bytes_val = float(bytes_val)

    if bytes_val < 1024:
        return f"{bytes_val:.0f}B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f}KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f}MB"
    elif bytes_val < 1024**4:
        return f"{bytes_val/1024**3:.1f}GB"
    else:
        return f"{bytes_val/1024**4:.1f}TB"


def add_region_labels(df):
    name_to_label_map = {
        "chr16p~15M": "16p1",
        "chr16p22M~28M": "16p2",
        "chr17p": "17p",
        "chr17p18M": "17p1",
        "chr17q~36M": "17q1",
        "chr17q36M~38M": "17q2",
        "chr17q38M~45M": "17q3",
        "chr17q45M~83M": "17q4",
        "chr18p": "18p",
        "chr19p": "19p",
        "chr20p": "20p",
        "chr20q31M~64M": "20q1",
        "chr20q": "20q",
        "chr21q34M~43M": "21q1",
        "chr21q44M~47M": "21q2",
        "chr22q16M~18M": "22q1",
        "chr22q19M~21M": "22q2",
        "Total": "Total",
    }
    df["region_name"] = df.region_name.map(name_to_label_map)
    return df.sort_values(by="region_name")


def create_regions_table(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df = add_region_labels(df)

    name_mapping = {
        "region_name": "region",
        "first_site_position": "start",
        "last_site_position": "end",
        "sequence_length": "length",
        "genic_proportion": "gene content",
        "recomb_rate_cM_Mb": "recomb rate",
    }

    # Format number columns
    number_columns = [
        "first_site_position",
        "last_site_position",
        "sequence_length",
    ]
    for col in number_columns:
        if col in df.columns:
            df[col] = df[col].apply(format_number)

    df = df.rename(columns=name_mapping)

    latex_table = df.to_latex(
        index=False,
        escape=False,
        float_format="%.3f",
        column_format="l" + "r" * (len(df.columns) - 1),
        label="tab:inference_regions",
    )
    return latex_table


def create_timing_table(csv_file_path):
    """Create LaTeX table for timing information"""
    df = pd.read_csv(csv_file_path)
    df = add_region_labels(df)

    # Select timing columns
    timing_columns = [
        "region_name",
        "generate_ancestors",
        "match_ancestors",
        "match_samples",
        "tsdate",
        "total_time",
    ]
    timing_df = df[timing_columns].copy()

    # Format timing columns
    time_columns = [
        "generate_ancestors",
        "match_ancestors",
        "match_samples",
        "tsdate",
        "total_time",
    ]
    for col in time_columns:
        if col in timing_df.columns:
            timing_df[col] = timing_df[col].apply(format_time)

    # Column mapping for timing table
    timing_column_mapping = {
        "region_name": "region",
        "generate_ancestors": "gen\_ancestors",
        "match_ancestors": "match\_ancestors",
        "match_samples": "match\_samples",
        "tsdate": "tsdate",
        "total_time": "total",
    }

    timing_df = timing_df.rename(columns=timing_column_mapping)

    latex_table = timing_df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (len(timing_df.columns) - 1),
        caption="Timing Results by Region",
        label="tab:timing_results",
    )
    return latex_table


def create_statistics_table(csv_file_path):
    """Create LaTeX table for dataset statistics"""
    df = pd.read_csv(csv_file_path)
    df = add_region_labels(df)

    # Select statistics columns
    stats_columns = [
        "region_name",
        "sequence_length",
        "number_of_sites",
        "number_of_trees",
        "number_of_nodes",
        "number_of_edges",
        "number_of_mutations",
        "tszip_total_size_bytes",
    ]
    stats_df = df[stats_columns].copy()

    # Format number columns
    number_columns = [
        "first_site_position",
        "last_site_position",
        "sequence_length",
        "number_of_sites",
        "number_of_samples",
        "number_of_trees",
        "number_of_nodes",
        "number_of_edges",
        "number_of_mutations",
    ]
    for col in number_columns:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].apply(format_number)

    # Format bytes column
    stats_df["tszip_total_size_bytes"] = stats_df["tszip_total_size_bytes"].apply(
        format_bytes
    )

    # Column mapping for statistics table
    stats_column_mapping = {
        "region_name": "region",
        "sequence_length": "length (bp)",
        "number_of_sites": "sites",
        "number_of_trees": "trees",
        "number_of_nodes": "nodes",
        "number_of_edges": "edges",
        "number_of_mutations": "muts",
        "tszip_total_size_bytes": "file size",
    }

    stats_df = stats_df.rename(columns=stats_column_mapping)

    latex_table = stats_df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (len(stats_df.columns) - 1),
        caption="Dataset Statistics by Region",
        label="tab:dataset_statistics",
    )

    return latex_table


if __name__ == "__main__":
    gel_csv_file_path = "data/inference-stats-and-timing.csv"

    # df = pd.read_csv(gel_csv_file_path)
    # dfs = df[
    #         [
    #             "region_name",
    #             "first_site_position",
    #             "last_site_position",
    #             "sequence_length",
    #         ]
    #     ]
    # dfs.to_csv("data/inference_regions.csv", index=False)

    table = create_regions_table("data/inference_regions.csv")
    print(table)

    timing_table = create_timing_table(gel_csv_file_path)
    gel_stats_table = create_statistics_table(gel_csv_file_path)
    print(gel_stats_table)

    tgp_csv_file_path = "data/tgp/inference_stats.csv"
    tgp_stats_table = create_statistics_table(tgp_csv_file_path)
    print(tgp_stats_table)
