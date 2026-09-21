"""Create the thesis figures from the verified CSV result tables."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_event_state"
IMAGES = ROOT / "images"


def save(figure, name: str) -> None:
    """Save a figure beside the result tables and into images/ for the thesis."""
    for folder in [RESULTS, IMAGES]:
        folder.mkdir(parents=True, exist_ok=True)
        figure.savefig(folder / name, dpi=220)


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
    "learnable": "whole system",
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
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    grouped_bars(
        axes[0],
        labels,
        [(name, table[f"{key}_event_energy_mae"], color) for name, key, color in decoders],
        "Attribution error per held-out event (log scale)",
        log=True,
    )
    for position, value in enumerate(floor):
        axes[0].hlines(value, position - 0.45, position + 0.45, colors="#333333",
                       linestyles="--", linewidth=1.1, zorder=5,
                       label="floor of the measure" if position == 0 else None)
    axes[0].set_ylim(0.1, 60)
    axes[0].set_title("Energy attributed to hidden events (lower is better)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    grouped_bars(
        axes[1],
        labels,
        [(name, 100 * table[f"{key}_state_accuracy"], color) for name, key, color in decoders],
        "Held-out events named correctly (%)",
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Activities named correctly (higher is better)")
    figure.tight_layout()
    save(figure, "rq1_rq3_overview.png")
    plt.close(figure)


def plot_rq2() -> None:
    """Costs and named activities as cases are compressed, median over the noise draws.

    The attribution errors of the decoders are given in the thesis table, so the
    right panel shows what the table does not: how many held-out activities each
    decoder still names correctly.
    """
    table = pd.read_csv(RESULTS / "rq2_correlation.csv")
    table = table.groupby("span_minutes", sort=False).median(numeric_only=True)
    labels = ["7 days", "2 hours", "30 minutes", "zero"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    grouped_bars(
        axes[0],
        labels,
        [
            ("Ordinary least squares", table["ols_mean_cost_error"], "#A0CBE8"),
            ("Weighted estimator", table["weighted_regression_mean_cost_error"], "#4E79A7"),
        ],
        "Activity cost error (log scale)",
        log=True,
    )
    axes[0].set_title("Activity costs (lower is better)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    grouped_bars(
        axes[1],
        labels,
        [
            ("No-transition control", 100 * table["independent_decoder_state_accuracy"], "#40B0A6"),
            ("Pooled decoder", 100 * table["pooled_decoder_state_accuracy"], "#6874E8"),
            ("Variant-first decoder", 100 * table["variant_decoder_state_accuracy"], "#F28E2B"),
        ],
        "Held-out events named correctly (%)",
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Activities named correctly (higher is better)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    for axis in axes:
        axis.set_xlabel("Case span after compression")
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
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    grouped_bars(
        axes[0],
        labels,
        [
            ("No-transition control", table["independent_decoder_event_energy_mae"], "#40B0A6"),
            ("Variant-first decoder", table["variant_decoder_event_energy_mae"], "#F28E2B"),
            ("Hybrid with OLS costs", table["variant_hmm_lrm_event_energy_mae"], "#8C564B"),
        ],
        "Attribution error per held-out event (log scale)",
        log=True,
    )
    axes[0].set_ylim(0.1, 60)
    axes[0].set_title("Energy attributed to hidden events (lower is better)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    grouped_bars(
        axes[1],
        labels,
        [
            ("No-transition control", table["independent_decoder_test_rmse"], "#40B0A6"),
            ("Variant-first decoder", table["variant_decoder_test_rmse"], "#F28E2B"),
            ("Hybrid with OLS costs", table["variant_hmm_lrm_test_rmse"], "#8C564B"),
        ],
        "Held-out RMSE (log scale)",
        log=True,
    )
    axes[1].set_ylim(1, 1000)
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
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    data, positions, colours = [], [], []
    for index, scope in enumerate(scopes):
        part = table[table["scope"] == scope]
        data += [part["ols_mean_cost_error"].to_numpy(),
                 part["weighted_regression_mean_cost_error"].to_numpy()]
        positions += [index - 0.18, index + 0.18]
        colours += ["#5D7CA6", "#F28E2B"]
        wins = int((part["weighted_regression_mean_cost_error"]
                    < part["ols_mean_cost_error"]).sum())
        axis.text(index, 0.97, f"weighted lower in {wins} of {len(part)} draws",
                  transform=axis.get_xaxis_transform(), ha="center", va="top",
                  fontsize=9, color="#333333")

    boxes = axis.boxplot(data, positions=positions, widths=0.28, patch_artist=True,
                         medianprops={"color": "black"}, flierprops={"markersize": 3})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    axis.set_xticks(range(len(scopes)), [SCOPE_LABELS[scope] for scope in scopes])
    axis.set_yscale("log")
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top * 3.0)
    axis.set_ylabel("Activity cost error over 30 noise draws (log scale)")
    axis.set_title("Cost recovery draw by draw (lower is better)")
    axis.grid(axis="y", alpha=0.25, which="both")
    axis.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.75)
                         for c in ["#5D7CA6", "#F28E2B"]],
                labels=["Ordinary least squares", "Weighted estimator"],
                frameon=False, fontsize=9, loc="lower right")
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
    names = {10080.0: "7 days", 120.0: "2 hours", 30.0: "30 minutes", 0.0: "zero"}
    labels = [names.get(span, f"{span:g} min") for span in median.index]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    grouped_bars(
        axes[0],
        labels,
        [("Ordinary least squares", median["ols_mean_cost_error"], "#5D7CA6"),
         ("Weighted estimator", median["weighted_regression_mean_cost_error"], "#F28E2B")],
        "Activity cost error (log scale)",
        log=True,
    )
    axes[0].set_title("The costs degrade as the cases are squeezed")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    grouped_bars(
        axes[1],
        labels,
        [("Ordinary least squares", median["ols_known_x_test_rmse"], "#5D7CA6"),
         ("Weighted estimator", median["weighted_regression_test_rmse"], "#F28E2B")],
        "Held-out RMSE with known activities",
    )
    axes[1].set_ylim(3.0, 4.2)
    axes[1].set_title("The rebuilt meter hardly moves")
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
