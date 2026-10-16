import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figures(summary, rankings, output):
    """Render the notebook's two journal figures from the supplied results."""
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "figure.titlesize": 14, "font.family": "sans-serif"})
    size = int(summary["Sample_Size"].max())
    rows = summary[summary["Sample_Size"] == size]
    grand = rankings[rankings["Sample_Size"] == size].sort_values("Grand_Multiplier", ascending=False)
    models = grand["Model"].tolist()
    domains = sorted(rows["Domain"].unique())
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    x, width = np.arange(len(domains)), 0.15
    for index, model in enumerate(models):
        values = rows[rows["Model"] == model].set_index("Domain").loc[domains, "Disparity_Multiplier"].to_numpy()
        offset = (index - len(models) / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width, label=model, color=colors[index % len(colors)],
                      edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, padding=3, fmt="%.2f", fontsize=8, weight="bold")
    actual_total = int(rows.groupby("Domain")["Pairs"].first().sum())
    ax.set_ylabel("Compute Disparity Multiplier (BN tokens / EN tokens)", weight="bold")
    ax.set_title(f"Figure 1: Empirical Tokenization Disparity Matrix Across Cross-Linguistic Domains (N={actual_total:,})",
                 weight="bold", pad=65)
    ax.set_xticks(x, domains, weight="bold")
    ax.set_ylim(0, max(6.0, float(rows["Disparity_Multiplier"].max()) * 1.2))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "fig1_cross_domain_disparity.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "fig1_cross_domain_disparity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    positions = np.arange(len(models))
    bars = ax.barh(positions, grand["Grand_Multiplier"], height=0.55,
                   color=["#c44e52", "#da8c60", "#55a868", "#4c72b0", "#8172b3"][:len(models)],
                   edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, padding=5, fmt=r"%.2f$\times$", fontsize=10, weight="bold")
    ax.set_yticks(positions, models, weight="bold")
    ax.invert_yaxis()
    ax.set_xlabel(r"Grand Arithmetic Mean Multiplier ($\mathcal{M}_{grand}$)", weight="bold")
    ax.set_title("Figure 2: 2026 Grand Open-Weight Tokenization Inefficiency Index (Worst to Best)", weight="bold", pad=15)
    ax.set_xlim(0, max(5.5, float(grand["Grand_Multiplier"].max()) * 1.2))
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1, label=r"Equal token-count baseline ($\mathcal{M}=1.0$)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "fig2_grand_inefficiency_ranking.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "fig2_grand_inefficiency_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
