"""Create the thesis figures from the verified CSV result tables.

Every figure is drawn at the width it is printed at, nine tenths of the text
width of the thesis, so that LaTeX does not scale it down and its labels stay as
readable on paper as the text around them. Panels are stacked rather than set
side by side for the same reason: a stacked panel keeps the full width.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_event_state"
IMAGES = ROOT / "images"

# Nine tenths of the thesis text width, in inches, and type sizes that survive
# printing at that size.
WIDTH = 5.58
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def save(figure, name: str) -> None:
    """Save a figure beside the result tables and into images/ for the thesis."""
    for folder in [RESULTS, IMAGES]:
        folder.mkdir(parents=True, exist_ok=True)
        figure.savefig(folder / name, dpi=220, bbox_inches="tight")


def grouped_bars(axis, labels, series, ylabel, log=False):
    """Draw readable side-by-side bars for several methods."""
    x = np.arange(len(labels))
    width = 0.78 / len(series)
    for index, (name, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        axis.bar(x + offset, values, width, label=name, color=color)
    axis.set_xticks(x, labels)
    axis.set_ylabel(ylabel)
    if log:
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)


SCOPE_LABELS = {
    "top1": "top one",
    "top2": "top two",
    "top3": "top three",
    "top5": "top five",
    "learnable": "whole\nsystem",
}


def plot_main() -> None:
    """Attribution of hidden events by all four decoders, against the floor of the measure.

    The cost and reconstruction comparisons have their own tables in the thesis,
    with the medians over thirty draws, so this figure shows only what those
    tables do not: how each decoder attributes energy and names activities once
    the activity labels are hidden. The floor is the attribution error left when
    every activity and every cost is exactly right, from thesis_facts.csv.
    """
    table = pd.read_csv(RESULTS / "event_state_results.csv")
    facts = pd.read_csv(RESULTS / "thesis_facts.csv").set_index("fact")["value"]
    labels = table["scope"].map(SCOPE_LABELS).tolist()
    floor = [float(facts[f"{scope}:attribution_error_floor"]) for scope in table["scope"]]
    decoders = [
        ("Position-only baseline", "position", "#BAB0AC"),
        ("No-transition control", "independent_decoder", "#40B0A6"),
        ("Pooled decoder", "pooled_decoder", "#6874E8"),
        ("Variant-first decoder", "variant_decoder", "#F28E2B"),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.95))

    grouped_bars(
        axes[0],
        labels,
        [(name, table[f"{key}_event_energy_mae"], color) for name, key, color in decoders],
        "Attribution error (log)",
        log=True,
    )
    for position, value in enumerate(floor):
        axes[0].hlines(value, position - 0.45, position + 0.45, colors="#333333",
                       linestyles="--", linewidth=1.1, zorder=5,
                       label="floor of the measure" if position == 0 else None)
    axes[0].set_ylim(0.1, 600)
    axes[0].set_title("Energy attributed to hidden events (lower is better)")
    axes[0].legend(frameon=False, loc="upper left", ncol=2, handlelength=1.3,
                   columnspacing=1.0)

    grouped_bars(
        axes[1],
        labels,
        [(name, 100 * table[f"{key}_state_accuracy"], color) for name, key, color in decoders],
        "Named correctly (%)",
    )
    axes[1].set_ylim(0, 118)
    axes[1].set_title("Activities named correctly (higher is better)")
    figure.tight_layout()
    save(figure, "rq1_rq3_overview.png")
    plt.close(figure)


def plot_rq2() -> None:
    """Costs and named activities as cases are compressed, median over the noise draws.

    The attribution errors of the decoders are given in the thesis table, so the
    lower panel shows what the table does not: how many held-out activities each
    decoder still names correctly.
    """
    table = pd.read_csv(RESULTS / "rq2_correlation.csv")
    table = table.groupby("span_minutes", sort=False).median(numeric_only=True)
    labels = ["7 days", "2 hours", "30 min", "zero"]
    figure, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.75))
    grouped_bars(
        axes[0],
        labels,
        [
            ("Ordinary least squares", table["ols_mean_cost_error"], "#A0CBE8"),
            ("Weighted estimator", table["weighted_regression_mean_cost_error"], "#4E79A7"),
        ],
        "Activity cost error (log)",
        log=True,
    )
    axes[0].set_ylim(1e-3, 30)
    axes[0].set_title("Activity costs (lower is better)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.3)
    grouped_bars(
        axes[1],
        labels,
        [
            ("No-transition control", 100 * table["independent_decoder_state_accuracy"], "#40B0A6"),
            ("Pooled decoder", 100 * table["pooled_decoder_state_accuracy"], "#6874E8"),
            ("Variant-first decoder", 100 * table["variant_decoder_state_accuracy"], "#F28E2B"),
        ],
        "Named correctly (%)",
    )
    axes[1].set_ylim(0, 128)
    axes[1].set_title("Activities named correctly (higher is better)")
    axes[1].legend(frameon=False, loc="upper right", handlelength=1.3)
    axes[1].set_xlabel("Case span after compression")
    figure.tight_layout()
    save(figure, "rq2_correlation.png")
    plt.close(figure)


def plot_rq4() -> None:
    """The variant-first decoder against the hybrid that charges its paths with OLS costs.

    Both panels use a logarithmic scale, because the narrow scopes sit near the
    floor of each measure while the whole system is two orders of magnitude
    higher, and a linear scale would flatten the narrow scopes to nothing.
    """
    table = pd.read_csv(RESULTS / "event_state_results.csv")
    labels = table["scope"].map(SCOPE_LABELS).tolist()
    series = [
        ("No-transition control", "independent_decoder", "#40B0A6"),
        ("Variant-first decoder", "variant_decoder", "#F28E2B"),
        ("Hybrid with OLS costs", "variant_hmm_lrm", "#8C564B"),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.75))
    grouped_bars(
        axes[0],
        labels,
        [(name, table[f"{key}_event_energy_mae"], color) for name, key, color in series],
        "Attribution error (log)",
        log=True,
    )
    axes[0].set_ylim(0.1, 400)
    axes[0].set_title("Energy attributed to hidden events (lower is better)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.3)
    grouped_bars(
        axes[1],
        labels,
        [(name, table[f"{key}_test_rmse"], color) for name, key, color in series],
        "Held-out RMSE (log)",
        log=True,
    )
    axes[1].set_ylim(1, 8000)
    axes[1].set_title("Meter rebuilt from decoded activities (lower is better)")
    figure.tight_layout()
    save(figure, "rq4_composition.png")
    plt.close(figure)


def plot_cost_draws() -> None:
    """The thirty noise draws behind the activity cost comparison.

    On the whole system the difference between the estimators is smaller than
    the swing between draws, so the thesis reports medians and win counts rather
    than one run. This figure shows the draws themselves: each box holds the
    thirty cost errors of one estimator on one scope, and the win count says how
    often the weighted estimator was the lower of the two within a draw.
    """
    table = pd.read_csv(RESULTS / "reward_estimator_seed_check.csv")
    scopes = [scope for scope in SCOPE_LABELS if scope in set(table["scope"])]
    figure, axis = plt.subplots(figsize=(WIDTH, 4.15))
    data, positions, colours = [], [], []
    for index, scope in enumerate(scopes):
        part = table[table["scope"] == scope]
        data += [part["ols_mean_cost_error"].to_numpy(),
                 part["weighted_regression_mean_cost_error"].to_numpy()]
        positions += [index - 0.18, index + 0.18]
        colours += ["#A0CBE8", "#F28E2B"]
        wins = int((part["weighted_regression_mean_cost_error"]
                    < part["ols_mean_cost_error"]).sum())
        axis.text(index, 0.98, f"weighted lower\nin {wins} of {len(part)}",
                  transform=axis.get_xaxis_transform(), ha="center", va="top",
                  fontsize=10, color="#333333", linespacing=1.2)

    boxes = axis.boxplot(data, positions=positions, widths=0.28, patch_artist=True,
                         medianprops={"color": "black"}, flierprops={"markersize": 3})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    axis.set_xticks(range(len(scopes)),
                    [SCOPE_LABELS[scope].replace("\n", " ") for scope in scopes])
    axis.set_yscale("log")
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top * 25.0)
    axis.set_ylabel("Cost error over 30 draws (log)")
    axis.set_title("Cost recovery draw by draw (lower is better)")
    axis.grid(axis="y", alpha=0.25, which="both")
    axis.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.85)
                         for c in ["#A0CBE8", "#F28E2B"]],
                labels=["Ordinary least squares", "Weighted estimator"],
                frameon=False, loc="lower right", handlelength=1.3)
    figure.tight_layout()
    save(figure, "cost_draws.png")
    plt.close(figure)


def plot_correlation_costs() -> None:
    """What compression does to the costs, and what it does to the meter.

    The stress test squeezes every case into a shorter span. The costs of both
    estimators degrade by orders of magnitude, while the meter rebuilt from
    those costs with known activities hardly moves, which is the clearest case
    in the thesis of a close fit hiding wrong costs.
    """
    table = pd.read_csv(RESULTS / "rq2_correlation.csv")
    median = table.groupby("span_minutes").median(numeric_only=True).sort_index(ascending=False)
    names = {10080.0: "7 days", 120.0: "2 hours", 30.0: "30 min", 0.0: "zero"}
    labels = [names.get(span, f"{span:g} min") for span in median.index]

    figure, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.6))
    grouped_bars(
        axes[0],
        labels,
        [("Ordinary least squares", median["ols_mean_cost_error"], "#A0CBE8"),
         ("Weighted estimator", median["weighted_regression_mean_cost_error"], "#F28E2B")],
        "Cost error (log)",
        log=True,
    )
    axes[0].set_ylim(1e-3, 30)
    axes[0].set_title("The costs degrade as the cases are squeezed")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.3)
    grouped_bars(
        axes[1],
        labels,
        [("Ordinary least squares", median["ols_known_x_test_rmse"], "#A0CBE8"),
         ("Weighted estimator", median["weighted_regression_test_rmse"], "#F28E2B")],
        "Held-out RMSE",
    )
    axes[1].set_ylim(3.0, 4.2)
    axes[1].set_title("The rebuilt meter hardly moves")
    axes[1].set_xlabel("Case span after compression")
    figure.tight_layout()
    save(figure, "correlation_costs.png")
    plt.close(figure)


if __name__ == "__main__":
    plot_main()
    plot_rq2()
    plot_rq4()
    plot_cost_draws()
    plot_correlation_costs()
    print(f"Saved figures to {RESULTS}")
