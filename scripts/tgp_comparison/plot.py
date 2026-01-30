import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
import argparse

def plot_method_hexplot(ax, df, method_x, method_y, shared_pos,
                    cmap="viridis", cbar_pad=0.02):
    """
    Plot a hexplot of mutation ages for two given methods at a provided array of 
    sites. 
    """
    
    pivot_df = df.pivot_table(index="pos_hg38", columns="method", values="midpoint_age").reset_index()
    pivot_df.dropna(subset=[method_x, method_y], inplace=True)

    x = pivot_df[method_x].astype(float).to_numpy()
    y = pivot_df[method_y].astype(float).to_numpy()
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    assert np.sum(m) > 0

    x, y = x[m], y[m]
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    r, _ = pearsonr(lx, ly)
    hb = ax.hexbin(x, y, xscale="log", yscale="log", mincnt=1, gridsize=80, cmap=cmap)
    
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    y_fit = 10 ** (intercept + slope * np.log10(x_fit))
    ax.plot(x_fit, y_fit, linewidth=2, color="firebrick", linestyle="--")
    ax.text(0.03, 0.97, f"$r = {r:.3f}$", ha="left", va="top", transform=ax.transAxes,
            fontsize=14, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = ax.figure.colorbar(hb, ax=ax, pad=cbar_pad, fraction=0.046)
    cbar.ax.set_title("Count", fontsize=10, pad=2.5)

def find_shared_positions(df):
    """
    Determine all the sites present across all methods in the dataframe.
    Where possible, only counts the sites with matching ancestral and 
    derived alleles. 
    """
    df_no_singer = df[df.method != "singer"]
    df_singer = df[df.method == "singer"]

    grouped = df_no_singer.groupby("method")["variant"].apply(set)
    shared_var = sorted(set.intersection(*grouped))
    shared_pos = df_no_singer[df_no_singer.variant.isin(shared_var)]["pos_hg38"]
    singer_pos = df_singer.pos_hg38
    shared_pos = sorted(set(shared_pos).intersection(set(singer_pos)))
    return shared_pos

def plot_hexbin_grid(input,
                     output,
                     method_dict=None,
                     cbar_pad=0.05,
                     plot_pad=0.23,
                     label_pad=0.015,
                     cmap="viridis"):

    if method_dict is None:
        method_dict = {
            "tsdate_ep": "tsinfer+tsdate",
            "coalNN": "CoalNN",
            "Relate": "Relate",
            "GEVA": "GEVA",
            "singer": "SINGER",
        }

    df = pd.read_csv(input)
    methods = list(method_dict.keys())
    n = len(methods)

    fig, axs = plt.subplots(n, n, figsize=(4.5 * n, 4.5 * n), sharex="col", sharey="row")
    if n == 1:
        axs = np.array([[axs]])

    df = df[df.method.isin(methods)]
    assert len(df) > 0
    shared_pos = find_shared_positions(df)
    print(f"{len(shared_pos)} sites are shared across methods")

    for i in range(n):
        for j in range(n):
            plot_method_hexplot(
                ax=axs[i, j],
                df=df,
                method_x=methods[j],
                method_y=methods[i],
                cmap=cmap,
                cbar_pad=cbar_pad,
                shared_pos=shared_pos,
            )
    plt.subplots_adjust(left=0.12, bottom=0.12, wspace=plot_pad, hspace=plot_pad)

    for i in range(n):
        ax_left = axs[i, 0]
        pos = ax_left.get_position()
        fig.text(pos.x0 - label_pad, pos.y0 + pos.height / 2, method_dict[methods[i]],
                 ha="right", va="center", rotation=90, fontsize=16)

        ax_bottom = axs[-1, i]
        pos = ax_bottom.get_position()
        fig.text(pos.x0 + pos.width / 2, pos.y0 - label_pad, method_dict[methods[i]],
                 ha="center", va="top", fontsize=16)

        ax_top = axs[0, i]
        pos = ax_top.get_position()
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.3 * label_pad, method_dict[methods[i]],
                 ha="center", va="bottom", fontsize=16)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output PNG file")
    parser.add_argument("--cbar_pad", help="Padding width to the left of colour legend", type=float, default=0.05)
    parser.add_argument("--plot_pad", help="Padding width between each plot", type=float, default=0.23)
    parser.add_argument("--label_pad", help="Padding between method labels and grid", type=float, default=0.015)
    parser.add_argument("--cmap", help="Colour map name", type=str, default="viridis")
    args = parser.parse_args()

    plot_hexbin_grid(args.input,
                     args.output,
                     cbar_pad=args.cbar_pad,
                     plot_pad=args.plot_pad,
                     label_pad=args.label_pad,
                     cmap=args.cmap)

