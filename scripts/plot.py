# Generate figure plots

import argparse
import collections
import json
import os.path
import string
import re

from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import weighted
from matplotlib.cbook import violin_stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
import numpy as np
import pandas as pd
import tskit
import tszip

top_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(top_dir, "data")

def read_csv(datafile):
    return pd.read_csv(os.path.join(data_dir, datafile), sep="\t")

def plot_sampling_sim(datafiles, fig=None, sel=None):
    assert len(datafiles) == 1
    datafile = os.path.join(data_dir, list(datafiles.values())[0])


    df = read_csv(datafile)
    # Get some info out of the column headers
    pop_data = [c for c in df.columns if c.startswith("pop=")]
    assert len(pop_data) == 1
    pop_col = pop_data[0]
    pop_data = json.loads(pop_col.replace("pop=", ""))
    sampling_data = [c for c in df.columns if c.startswith("sampling_scheme=")]
    assert len(sampling_data) == 1
    sampling_col = sampling_data[0]
    sampling_data = json.loads(sampling_col.replace("sampling_scheme=", ""))

    # slight hack to get the sample size
    sample_sizes = {sum(samp.values()) for samp in sampling_data.values()}
    assert len(sample_sizes) == 1
    sample_size = sample_sizes.pop()

    print(f"Decompressed: plotting sampling simulation from {datafile} with {sample_size} samples")
    # zeroth col is "b" (balanced) or "u" (unbalanced)
    data = {
        sampling: df[df[sampling_col] == sampling[0]]
        for sampling in sorted(sampling_data.keys())
    }

    _, axes = plt.subplots(
        len(data), 4,
        figsize=(16, 6 * len(data)),
        sharex="col",
        squeeze=False,
        gridspec_kw={'width_ratios': [1, 3, 0.2, 3], 'wspace': 0.1, 'hspace': 0.07}
    )
    linestyle = {
        "YRI": {"ls":"solid", "c": "black"},
        "CEU": {"ls":"dashed", "c": "red"},
        "CHB": {"ls":"dotted", "c": "cyan"}
    }
    pop_time = {"CEU": 5600, "YRI": 100_000, "CHB": 848}
    long_name = {"CEU": "Europe", "YRI": "Africa", "CHB": "Asia"}

    for axs, sim, label in zip(axes, data.keys(), ('(i)', '(ii)', '(iii)')):
        gens = np.array([1, 10, 100, 1000, 10000, 100000])
        freqs = np.array([0.001, 0.01, 0.1, 1])
        sample_props = [f"{sampling_data[sim][nm]//2} diploids\nfrom {long_name[nm]}\n" for nm in linestyle.keys()]
        axs[0].text(
            0.2, 0.8, label, size=20, ha="center", va="center"
        )
        axs[0].text(
            0.2,
            0.5,
            ("$\\bf {" + sim.capitalize() + "}$" +
            "\n$\\bf{sampling}$\n\n") + "\n".join(sample_props),
            size=12,
            ha="center",
            va="center",
        )
        axs[0].axis("off")

        # Add a generation of 1 to the log values so that <1 is not over-exaggerated
        log10true = np.log10(data[sim]["true_midtime"] + 1)
        log10infer = np.log10(data[sim]["inferred_time"] + 1)
        log10freq = np.log10(data[sim]["freq"])
        axs[1].hexbin(log10true, log10freq, bins="log", cmap='plasma', gridsize=80)
        axs[1].set_ylabel("Derived allele frequency in whole dataset")
        axs[1].set_yticks(np.log10(freqs), labels=freqs, rotation=90, va='center')
        axs[1].set_xticks(np.log10(gens+1), labels=gens)
        slices = np.log10(np.logspace(np.log10(1/sample_size), 0, 20))
        for name, params in linestyle.items():
            line_x = []
            line_y = []
            for lo, hi in zip(slices[:-1], slices[1:]):
                use = np.logical_and.reduce(
                    (log10freq >= lo, log10freq < hi, data[sim][pop_col]==pop_data[name])
                )
                line_x.append(np.mean(log10true[use]))     
                line_y.append((lo + hi) / 2)
                # don't draw the line once we are near the oldest time, as the mean suffers from truncation effects
                if line_x[-1] > np.log10(pop_time[name]/2):
                    break
            axs[1].plot(line_x, line_y, label=f"{long_name[name]}", **params)
        legend = axs[1].legend(loc="upper left", alignment="center")
        legend.set_title("Mean age of mutations\noriginating in:")  
        
        # Shim
        axs[2].set_visible(False)

        # Inferred times
        x = np.log10(data[sim]["true_midtime"])
        y = np.log10(data[sim]["inferred_time"])
        use = np.logical_and(np.isnan(x) == False, np.isnan(y) == False)
        x = x[use]
        y = y[use]
        axs[3].text(
            1,
            5,
            f"$r$={np.corrcoef(x, y)[0, 1]:.4f}\n$\\rho$={scipy.stats.spearmanr(x, y).statistic:.4f}",
            size=13,
            ha="center",
            va="bottom",
        )
        axs[3].axline(xy1=[1,1], slope=1, c="lightgrey")
        slices = np.log10(np.logspace(0, 5, 20))
        for name, params in linestyle.items():
            line_x = []
            line_y = []
            for lo, hi in zip(slices[:-1], slices[1:]):
                use = np.logical_and.reduce(
                    (log10true >= lo, log10true < hi, data[sim][pop_col]==pop_data[name])
                )
                line_x.append(np.mean(log10true[use]))     
                line_y.append((lo + hi) / 2)
                # don't draw the line once we are near the oldest time, as the mean suffers from truncation effects
                if line_x[-1] > np.log10(pop_time[name]/2):
                    break
            axs[3].plot(line_x, line_y, label=f"{long_name[name]}", **params)
        axs[3].hexbin(log10true, log10infer, bins="log", gridsize=80)
        axs[3].set_ylabel("Inferred mutation age (generations ago)")
        axs[3].set_xticks(np.log10(gens + 1), labels=gens)
        axs[3].set_yticks(np.log10(gens + 1), labels=gens, rotation=90, va='center')
        
        axins = axs[3].inset_axes([0.74, 0.02, 0.16, 0.35])
        hist = axins.hist(log10infer - log10true, orientation='horizontal', bins=50, alpha=0.5)
        axins.set_xlim(max(hist[0]) * 1.05, 0)
        axins.yaxis.tick_right()
        axins.set_yticks([-1, 0, 1])
        axins.set_yticklabels([-1, 0, 1], fontsize=7)
        axins.set_ylim(-1.3, 1.3)
        axins.yaxis.set_label_position("right")
        axins.set_ylabel("$\\log_{10}$(age) errors", labelpad=0)
        axins.axhline(y=0)
        axins.set_xticks([])



        if sim.startswith("unbalanced"):
            axs[1].set_xlabel("True age (mutation branch midpoint, generations ago)")
            axs[3].set_xlabel("True age (mutation branch midpoint, generations ago)")
        else:
            axs[1].set_title("a", size=20, loc='left')
            axs[3].set_title("b", size=20, loc='left')


    plt.savefig(os.path.join(top_dir, "figures", f"sampling_sim+{sample_size}.pdf"), bbox_inches='tight')


def pedigree_accuracy_plot(ax, df, full_df, inset_width=0.2):
    cutoffs = {}
    for c in df["cutoff"].unique():
        nmuts = df[df["cutoff"] == c]["nmuts"].values
        assert np.all(nmuts == nmuts[0])
        if np.isfinite(c):
            label = f"Subset of {nmuts[0]} recent muts\n(time < {c:g} generations)"
            finite_cutoff_val = c
        else:
            label = f"All {nmuts[0]} mutations"
        cutoffs[label] = c
    assert len(cutoffs) == 2
    ax.set_ylabel("Correlation coefficient ($r$)")
    # Make our own log axis, so that we can plot violin plots on the same scale
    ax.set_xlim(np.log10([7, 10e6]))
    ax.set_xticks(np.arange(1, 7), labels=[f"$10^{x}$" for x in np.arange(1, 7)])
    ax.set_xticks(np.log10(np.outer([2, 4, 6, 8], 10 ** np.arange(1, 7)).flatten()), minor=True)
    corr_y_pos = {}
    for label, c in cutoffs.items():
        data = df[np.logical_and(df["cutoff"] == c, df["method"] == "midpoint")]
        ax.plot(np.log10(data["sample_size"]), data["corr_coef"], "o-", label=label)
        corr_y_pos[c] = {k: v for k,v in zip(data["sample_size"], data["corr_coef"])}
        
    ax.legend()

    sample_sizes = df["sample_size"].unique()
    inset_params = ((0.03, sample_sizes[0]), (0.48, sample_sizes[-3]), (0.79, sample_sizes[-1]))
    y_pos_top = 0.83
    use_df = full_df[full_df[f"cutoff{finite_cutoff_val:g}"]]  # Only look at the ones with a cutoff
    for x_pos, nsamp in inset_params:
        inax = inset_axes(
            ax, width="100%", height="100%",
            bbox_to_anchor=(x_pos, y_pos_top - 0.3, inset_width, .3),
            bbox_transform=ax.transAxes
        )
        axcol = "tab:orange"
        inax.spines['bottom'].set_color(axcol)
        inax.spines['top'].set_color(axcol) 
        inax.spines['right'].set_color(axcol)
        inax.spines['left'].set_color(axcol)
        inax.tick_params(axis='both', which='both', colors=axcol)
        inax.set_xscale("log")
        inax.set_yscale("log")
        x, y = 10 ** use_df["true"], 10 ** use_df[str(nsamp)]

        inax.hexbin(x, y, xscale="log", yscale="log", bins="log")
        lpos = 10.0 ** np.array([-3, -1, 1, 3, 5])
        lminpos =  np.outer([20, 40, 60, 80], lpos[:-1]).flatten()
        inax.plot(lpos, lpos, c="k", alpha=0.2)
        lim = 0.05, 2e5
        for axis in (inax.xaxis, inax.yaxis):
            axis.set_ticks(lpos, labels=[""] * len(lpos))
            axis.set_ticks(lminpos, labels=[""] * len(lminpos), minor=True)
            axis.set_tick_params(which="both", direction="in")
        inax.set_xlim(*lim)
        inax.set_ylim(*lim)
    # Only show labels on the last one
    inax.set_xlabel("True midpoint time", size="xx-small", labelpad=1)
    inax.set_ylabel("Inferred midpoint time", size="xx-small", labelpad=1)
    
    for xfrac, nsamp in inset_params:
        (xl, yp), (xr, _) = (ax.transAxes + ax.transData.inverted()).transform(
            [(xfrac - 0.015, y_pos_top+0.025),
            (xfrac+inset_width - 0.015, y_pos_top+0.025)]     
        )
        triangle = plt.Polygon([
            [xl, yp],
            [np.log10(nsamp), corr_y_pos[finite_cutoff_val][nsamp]],
            [xr, yp]
        ], color="black", alpha=0.1)
        ax.add_patch(triangle)
    return ax

def pedigree_accuracy_distributions_plot(ax, full_df):
    def modify_line_segments(line_collection, length_ratio, side):
        # To shorten violinplot lines
        if line_collection is not None and hasattr(line_collection, 'get_segments'):
            # Get the segments of the lines
            segments = line_collection.get_segments()
            
            # Modify each segment
            new_segments = []
            for seg in segments:
                # Calculate positions and current width
                x_min, x_max = seg[:, 0].min(), seg[:, 0].max()
                x_len = x_max  -x_min
                # Create new segment with reduced width
                new_seg = seg.copy()
                new_seg[:, 0] = new_seg[:, 0] + (
                    [x_len * (1-length_ratio), 0] if side=="low" else [0, -x_len * (1-length_ratio)]
                )
                
                new_segments.append(new_seg)
            # Update the segments of the LineCollection
            line_collection.set_segments(new_segments)

    cutoffs = {None: np.ones(len(full_df), dtype=bool)}
    cutoffs.update({int(c.replace("cutoff", "")): full_df[c].values for c in full_df.columns if "cutoff" in c})
    assert len(cutoffs) == 2
    x = full_df["true"]
    y = {int(n): (full_df[n]-x) for n in full_df.columns if n.isdigit()}
    y = dict(sorted(y.items()))
    pos = np.log10(list(y.keys()))
    for side, use in zip(['low', 'high'], cutoffs.values()):
        parts = ax.violinplot(
            [v[use] for v in y.values()],
            positions=pos,
            widths=np.mean(np.diff(pos)) / 1.3,
            vert=True,
            showmedians=True,
            showextrema=True,
            quantiles=[[0.05, 0.95]] * len(y),
            side=side)
        if 'cquantiles' in parts:  # Shorten quantile lines
            modify_line_segments(parts['cquantiles'], 0.6, side=side)
        if 'cmaxes' in parts:  # Shorten extremum lines
            modify_line_segments(parts['cmaxes'], 0.3, side=side)
        if 'cmins' in parts:  # Shorten extremum lines
            modify_line_segments(parts['cmins'], 0.3, side=side)

    ax.axhline(y=0, alpha=0.2, c="black", ls=":")
    ax.set_ylabel("Mut. midpoint time error\n($\\log_{10}$inferred - $\\log_{10}$true)")
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])

def plot_pedigree_sim(datafiles, fig=None, sel=None):

    accuracy_df = read_csv(datafiles['ACCURACY_'])
    full_df = read_csv(datafiles['FULL_'])
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), height_ratios=(2, 1), sharex=True)
    axes[-1].set_xlabel("Number of sampled genomes ($n$)")
    # inset plots based on the last item in `cutoffs`
    pedigree_accuracy_plot(axes[0], accuracy_df, full_df)
    pedigree_accuracy_distributions_plot(axes[1], full_df)
    plt.savefig(os.path.join(top_dir, "figures", f"pedigree_accuracy.pdf"), bbox_inches='tight')


def validation_real_aDNA_plot(ax, aDNA_df, chrom):
    # create rng
    rng = np.random.default_rng(42)
    # normal distribution jitter
    jitter = rng.normal(1, 0.0075, size=len(aDNA_df))
    x = aDNA_df.AADR
    y = aDNA_df.tsdate_vgamma_upper
    plt.scatter(x * jitter, y, alpha=0.1, s=0.5)
    plt.xscale("log")
    plt.xlim(3e3)
    plt.xlabel("Min historical date (years before 1950)")
    plt.yscale("log")
    plt.ylabel("Max mutation age (generations ago)")
    plt.ylim(3e3/30)
    
    for generation_time, color in ((27, "tab:green"), (25, "tab:orange")):
        lnx = np.logspace(2, 6, 10)
        # convert from 1950 to 2010 (60 years), then to generations
        lny = (lnx + 60) / generation_time
        consistency = np.sum((x+ 60) / generation_time < y) /  len(x)
        
        old_y = aDNA_df.tsdate_inout_upper
        old_consistency = np.sum((x+ 60) / generation_time < old_y) /  len(x)

        label = (
            f"$\\bf{{Generation}}$ $\\bf{{time}}$ = $\\bf{{ {generation_time} }}$ $\\bf{{years}}$\n"
            f"  variational_gamma consistency: {consistency:.2%}\n"
            f"  inside_outside consistency: {old_consistency:.2%}"
        )
        plt.plot(lnx, lny, ls="-", label=label, color=color)
        #label this line "lower bound" along the diagonal
    ax.text(
            1.5e4, 1.5e4/30 - 100,
            f"Lower bound",
            rotation=15,
    )
    plt.legend(prop={'size': 9})
    print(f"Min aDNA date: {min(x[x!=0]) - 1950 + 2025} years BP, ",
          f"max aDNA date: {max(x) - 1950 + 2025} years BP"
          )


def validation_real_phlash_plot(ax, phlash_df, tsdate_df, chrom):
    colours = {
        "AFR": "tab:orange",
        "EUR": "tab:blue",
        "EAS": "tab:green",
    }
    for region in colours.keys():
        df = phlash_df[phlash_df["pop"] == region]
        ax.plot(
            df["gens"], 
            df["iicr"],
            label=region +  " phlash",
            c=colours[region],
            ls="--",
            alpha=0.5,
        )
    for region in colours.keys():
        df = tsdate_df[tsdate_df["pop"] == region]
        ax.plot(
            df["gens"], 
            df["iicr"],
            label=region +  f" tsdate (chr{chrom})",
            c=colours[region],
            alpha=0.5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(100, 1e4)
    ax.set_ylim(1e3, 2e5)
    ax.set_ylabel("Instantaneous Inverse Coalecence Rate ($N_e$)")
    ax.set_xlabel(f"Time (generations ago)")
    ax.legend()


def validation_real_chr17_inversion_plot(
    ax,
    coal_mat,
    genome_windows,
    time_windows,
    relate_df,
):
    min_time = time_windows[:-1].min()
    max_time = time_windows.max()
    left = genome_windows[1:].min()
    right = genome_windows[:-1].max()

    colormap = ax.pcolormesh(genome_windows, time_windows, coal_mat, cmap="viridis")
    for _, row in relate_df.iterrows():
        ax.hlines(
            y=row["time_high"],
            xmin=row["start_hg38"],
            xmax=row["end_hg38"],
            color="red",
            linewidth=4,
        )

    red_line = Line2D(
        [0], [0], color="red", linewidth=4, label="Upper bound\nage estimate\nfrom Relate"
    )
    ax.legend(handles=[red_line], loc="lower right", borderpad=1)
    plt.colorbar(colormap, ax=ax, label="Log10 coalescence rate")

    ax.set_xlim(left, right)
    xtickpos = np.arange(np.ceil(left / 1e5) * 1e5, np.ceil(right / 1e5) * 1e5, 1e5)
    ax.set_xticks(xtickpos, labels=xtickpos / 1e6)
    ax.set_ylim(min_time, max_time)
    ax.set_yscale("log")
    ax.set_xlabel("Position (Mb)\nChromosome 17")
    ax.set_ylabel("Time (generations ago)")


def plot_validation_aDNA(datafiles, fig=None, sel=None):
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax.set_title(f"Validating tsdate ages for chr{sel} against aDNA constraints")
    aDNA_df = read_csv(datafiles['TSDATE_'])
    validation_real_aDNA_plot(ax, aDNA_df, chrom=sel)
    plt.savefig(os.path.join(top_dir, "figures", f"{fig}.pdf"), bbox_inches='tight')

def plot_validation_OOA(datafiles, fig=None, sel=None):
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    #ax.set_title("Out-of-Africa timing vs. PHLASH")
    phlash_df = read_csv(datafiles['PHLASH_'])
    tsdate_df = read_csv(datafiles['TSDATE_'])
    validation_real_phlash_plot(ax, phlash_df, tsdate_df, chrom=sel)
    plt.savefig(os.path.join(top_dir, "figures", f"{fig}.pdf"), bbox_inches='tight')

def plot_validation_inversion(datafiles, fig=None, sel=None):
    """
    Plots the cross coalescence rates between carriers (H1 haplotype) and non-carriers
    (H2 haplotype) of the chr17q21.31 inversion. We the overlay estimated upper bounds
    for the inversion ages shown in Figure 7 of Ignatieva et al. (2024)
    """
    coal_mat = pd.read_csv(
        os.path.join(data_dir, datafiles["coal_mat_"]), header=None
    ).to_numpy()
    genome_windows = pd.read_csv(
        os.path.join(data_dir, datafiles["genome_windows_"]), header=None
    ).squeeze("columns").to_numpy()
    time_windows = pd.read_csv(
        os.path.join(data_dir, datafiles["time_windows_"]), header=None
    ).squeeze("columns").to_numpy()
    relate_df = read_csv(datafiles["relate_"])

    _, ax = plt.subplots(1, 1, figsize=(12, 6))
    #ax.set_title( "Cross coalescence rates between inversion carriers vs non-carriers")
    validation_real_chr17_inversion_plot(ax=ax,
                                         genome_windows=genome_windows,
                                         time_windows=time_windows,
                                         coal_mat=coal_mat,
                                         relate_df=relate_df)
    plt.savefig(os.path.join(top_dir, "figures", f"{fig}.pdf"), bbox_inches='tight')


def plot_mutation_subset(
    ax, full_times, subset_times, bin_width=0.05, xlabel=None, ylabel=None, title=None
):
    """
    Make a hexbin plot of mutation ages from a full and subset ARG, with an inset
    error histogram.
    """
    if len(full_times) != len(subset_times):
        raise ValueError(
            f"Length mismatch: {len(full_times)} full vs {len(subset_times)} subset times"
        )
    
    log_full = np.log10(full_times)
    log_subset = np.log10(subset_times)
    corr = np.corrcoef(log_subset, log_full)[0, 1]
    bias = np.mean(log_subset - log_full)
    mean_ft = full_times.mean()
    info = f"$r = {corr:.3f}$\n$\\mathrm{{bias}} = {bias:.3f}$"

    ax.hexbin(
        full_times, subset_times, xscale="log", yscale="log", mincnt=1, cmap="viridis"
    )
    ax.axline(
        (mean_ft, mean_ft),
        (mean_ft + 1, mean_ft + 1),
        linestyle="--",
        color="firebrick",
    )
    ax.text(0.01, 0.99, info, ha="left", va="top", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    log_resid = log_subset - log_full
    lo = np.floor(log_resid.min() / bin_width) * bin_width
    hi = np.ceil(log_resid.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)

    axins = ax.inset_axes([0.75, 0.02, 0.23, 0.38])
    counts, *_ = axins.hist(log_resid, bins=edges, orientation="horizontal", alpha=0.5)
    axins.set_xlim(max(counts) * 1.05, 0)
    axins.yaxis.tick_right()
    axins.set_yticks([-1, 0, 1])
    axins.set_ylim(-1.3, 1.3)
    axins.axhline(0, color="grey", linewidth=0.8)
    axins.set_xticks([])

    axins.add_patch(
        Rectangle(
            (-0.19, 0),
            0.19,
            1.0,
            transform=axins.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=-1,
            clip_on=False,
        )
    )
    axins.text(
        -0.02,
        0.5,
        "$\\log_{10}$(age) errors",
        rotation=90,
        va="center",
        ha="right",
        transform=axins.transAxes,
        size=8,
    )


def plot_tgp_singleton(datafiles, fig=None, sel=None, subset="1kgp_1500"):
    """
    Plot comparison of singleton methods using 1kgp data.
    """
    label_dict = {
        "random_phase": "Random phase",
        "phase_agnostic": "Phase-agnostic",
    }
    assert len(datafiles) == 1
    path = os.path.join(data_dir, list(datafiles.values())[0])
    df = pd.read_csv(path, sep="\t")
    df = df[df.subset == subset]
    full_times = df.loc[df.type == "true_phase", "time"].values
    figure, axs = plt.subplots(
        1, len(label_dict), figsize=(8, 4), sharey=True, sharex=True
    )
    for i, (var, label) in enumerate(label_dict.items()):
        ax = axs[i]
        subset_times = df.loc[df.type == var, "time"].values
        assert len(subset_times) > 0
        plot_mutation_subset(
            ax=ax,
            ylabel=f"Singleton age ({label.lower()})",
            full_times=full_times,
            subset_times=subset_times,
            title=label,
        )

    figure.text(0.5, 0.01, "Singleton age (true phase)", ha="center")
    plt.savefig(os.path.join(top_dir, "figures", f"{fig}.pdf"), bbox_inches="tight")


def plot_tgp_subset(datafiles, sel=None, fig=None):
    """
    Plot mutation age comparison across all frequencies for ARGs inferred from various
    subsets of 1kgp.
    """
    assert len(datafiles) == 1
    path = os.path.join(data_dir, list(datafiles.values())[0])
    df = pd.read_csv(path, sep="\t")
    full_times = df.loc[df.subset == "1kgp_all", "time"].values
    figure, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
    for i, subset in enumerate(["1kgp_100", "1kgp_300", "1kgp_1500"]):
        ax = axs[i]
        subset_times = df.loc[df.subset == subset, "time"].values
        subset_size = int(re.search(r"_(\d+)", subset).group(1))
        plot_mutation_subset(
            ylabel="Mutation age in subset ARG" if i == 0 else "",
            ax=ax,
            full_times=full_times,
            subset_times=subset_times,
            title=f"n = {subset_size}",
        )
    figure.text(0.5, 0.01, "Mutation age in all-sample ARG", ha="center")
    plt.savefig(os.path.join(top_dir, "figures", f"{fig}.pdf"), bbox_inches="tight")


choices = {
    "pedigree_sim": plot_pedigree_sim,
    "sampling_sim": plot_sampling_sim,
    "validation_aDNA": plot_validation_aDNA,
    "validation_OOA": plot_validation_OOA,
    "validation_inversion": plot_validation_inversion,
    "tgp_singleton": plot_tgp_singleton,
    "tgp_subset": plot_tgp_subset,
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
            "In the case of pedigree_sim, this is the number of samples. "
            "In the case of sampling_sim, this is the random seed used. "
            "In the case of validation_real it is the chromosome used. If not "
            "specified, the script will check there is only one valid set of files."
        ),
        default=None,  # default involves checking if there are any existing files
    )
    args = argparser.parse_args()
    fig = args.figure
    sel = args.select
    files = collections.defaultdict(dict)
    for fn in os.listdir(data_dir):
        if (m := re.match(fr"{fig}_(\w*)data\+(\d+).csv", fn)):
            key = int(m.group(2))
            files[key][m.group(1)] = fn  # e.g. "sampling_sim_DESC_data+123.csv"
    if sel is None:
        if len(files) == 1:
            sel = list(files.keys())[0]
        elif len(files) == 0:
            raise ValueError(f"No `{fig}_DESC_data+XXX.csv` files found in {data_dir}")
        else:
            raise ValueError(
                f"Multiple files (n={list(files.keys())}, please specify one via --select"
            )
    else:
        if sel not in files:
            raise ValueError(f"No `{fig}_DESC_data+{sel}.csv` files found in {data_dir}")

    choices[args.figure](datafiles=files[sel], fig=fig, sel=sel)
