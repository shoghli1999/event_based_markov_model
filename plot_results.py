"""Create the thesis figures from the verified CSV result tables."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_event_state"


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


def plot_main() -> None:
    """Separate reward estimation from the genuine transition comparison."""
    table = pd.read_csv(RESULTS / "event_state_results.csv")
    labels = table["scope"].replace({"learnable": "whole\nlearnable"}).tolist()
    colors = ["#6874E8", "#40B0A6", "#F28E2B", "#E15759"]
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.6))

    grouped_bars(
        axes[0],
        labels,
        [
            ("OLS", table["ols_known_x_test_rmse"], colors[0]),
            ("Ridge", table["ridge_known_x_test_rmse"], colors[1]),
            ("Weighted regression", table["weighted_regression_test_rmse"], colors[2]),
            ("Event-state HMM", table["hmm_known_x_test_rmse"], colors[3]),
        ],
        "Held-out RMSE",
    )
    axes[0].set_title("Known activities: reconstruction")
    axes[0].legend(frameon=False, fontsize=8)

    grouped_bars(
        axes[1],
        labels,
        [
            ("OLS", table["ols_mean_cost_error"], colors[0]),
            ("Ridge", table["ridge_mean_cost_error"], colors[1]),
            (
                "Weighted regression",
                table["weighted_regression_mean_cost_error"],
                colors[2],
            ),
            ("Event-state HMM", table["hmm_mean_cost_error"], colors[3]),
        ],
        "Mean activity-cost error (log scale)",
        log=True,
    )
    axes[1].set_title("Known activities: reward estimation")

    grouped_bars(
        axes[2],
        labels,
        [
            (
                "No transitions",
                table["independent_decoder_event_energy_mae"],
                "#BAB0AC",
            ),
            (
                "Pooled HMM",
                table["pooled_decoder_event_energy_mae"],
                colors[0],
            ),
            (
                "Variant-first HMM",
                table["variant_decoder_event_energy_mae"],
                colors[2],
            ),
        ],
        "Mean event-energy error",
    )
    axes[2].set_title("Hidden activities: attribution")
    axes[2].legend(frameon=False, fontsize=8)

    grouped_bars(
        axes[3],
        labels,
        [
            ("Position only", 100 * table["position_state_accuracy"], "#BAB0AC"),
            (
                "No transitions",
                100 * table["independent_decoder_state_accuracy"],
                colors[1],
            ),
            ("Pooled HMM", 100 * table["pooled_decoder_state_accuracy"], colors[0]),
            ("Variant-first HMM", 100 * table["variant_decoder_state_accuracy"], colors[2]),
        ],
        "Correctly decoded activities (%)",
    )
    axes[3].set_ylim(0, 105)
    axes[3].set_title("Hidden activities: state accuracy")
    axes[3].legend(frameon=False, fontsize=8)
    figure.suptitle("One model throughout: every hidden state is an activity", fontsize=14)
    figure.tight_layout()
    figure.savefig(RESULTS / "rq1_rq3_overview.png", dpi=220)
    plt.close(figure)


def plot_rq2() -> None:
    """Show reward and transition results as chains are compressed."""
    table = pd.read_csv(RESULTS / "rq2_correlation.csv")
    table = table.groupby("span_minutes", sort=False).median(numeric_only=True)
    labels = ["7 days", "2 hours", "30 min", "0 min"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    grouped_bars(
        axes[0],
        labels,
        [
            ("OLS", table["ols_mean_cost_error"], "#6874E8"),
            (
                "Weighted regression",
                table["weighted_regression_mean_cost_error"],
                "#F28E2B",
            ),
            ("Event-state HMM", table["hmm_mean_cost_error"], "#E15759"),
        ],
        "Mean activity-cost error (log scale)",
        log=True,
    )
    axes[0].set_title("Known-activity reward estimation")
    axes[0].legend(frameon=False)
    grouped_bars(
        axes[1],
        labels,
        [
            (
                "No transitions",
                table["independent_decoder_event_energy_mae"],
                "#40B0A6",
            ),
            (
                "Pooled HMM",
                table["pooled_decoder_event_energy_mae"],
                "#6874E8",
            ),
            (
                "Variant-first HMM",
                table["variant_decoder_event_energy_mae"],
                "#F28E2B",
            ),
        ],
        "Mean hidden-event energy error",
    )
    axes[1].set_title("Matched transition comparison")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("RQ2: correlation stress test")
    figure.tight_layout()
    figure.savefig(RESULTS / "rq2_correlation.png", dpi=220)
    plt.close(figure)


def plot_rq4() -> None:
    """Show the trade-off in the requested HMM and LRM composition."""
    table = pd.read_csv(RESULTS / "event_state_results.csv")
    labels = table["scope"].replace({"learnable": "whole\nlearnable"}).tolist()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    grouped_bars(
        axes[0],
        labels,
        [
            ("No transitions", table["independent_decoder_event_energy_mae"], "#40B0A6"),
            ("Variant HMM", table["variant_decoder_event_energy_mae"], "#F28E2B"),
            ("Variant HMM + LRM", table["variant_hmm_lrm_event_energy_mae"], "#6874E8"),
        ],
        "Mean hidden-event energy error",
    )
    axes[0].set_title("Event-level attribution")
    axes[0].legend(frameon=False, fontsize=8)
    grouped_bars(
        axes[1],
        labels,
        [
            ("No transitions", table["independent_decoder_test_rmse"], "#40B0A6"),
            ("Variant HMM", table["variant_decoder_test_rmse"], "#F28E2B"),
            ("Variant HMM + LRM", table["variant_hmm_lrm_test_rmse"], "#6874E8"),
        ],
        "Aggregate held-out RMSE",
    )
    axes[1].set_title("Meter reconstruction")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("RQ4: combining variant HMMs and linear rewards")
    figure.tight_layout()
    figure.savefig(RESULTS / "rq4_composition.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    plot_main()
    plot_rq2()
    plot_rq4()
    print(f"Saved figures to {RESULTS}")
