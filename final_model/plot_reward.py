"""
plot_reward.py
──────────────
One figure for the Markov reward layer, read from reward_layer.csv.

Left panel   the real test: how far the prediction is from the truth on cases
             the chain never saw. Training agreement is shown beside it only for
             comparison, because a chain built by counting always reproduces the
             cases it counted, so that bar cannot fail.
Middle panel the energy of one complete case on TRAINING cases, measured against
             predicted. The held-out comparison lives in the left panel; keeping
             them apart stops the two being read as one result.
Right panel  a consistency check: the per-variant chains and the pooled chain
             must expect the same number of events per case.

Usage
─────
    python final_model/plot_reward.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "results_event_state"
GREY, BLUE, GREEN, ORANGE = "#888888", "#4477AA", "#228833", "#EE7733"


def bars(axis, positions, series, ylabel, title):
    """Draw one group of bars per scope and label the axis."""
    width = 0.8 / len(series)
    for index, (label, values, colour) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        axis.bar(positions + offset, values, width, color=colour, label=label)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10, loc="left")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")


def label_bars(axis, positions, series, fmt="{:.2f}"):
    """Print each bar's value above it."""
    width = 0.8 / len(series)
    for index, (_, values, _) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        for position, value in zip(positions + offset, values):
            if not np.isfinite(value):
                continue
            axis.text(position, value, fmt.format(value),
                      ha="center", va="bottom", fontsize=7, color="dimgray")


def main():
    """Draw the reward-layer figure from the saved table."""
    table = pd.read_csv(OUT_DIR / "reward_layer.csv")
    scopes = table["scope"].tolist()
    x = np.arange(len(scopes))
    figure, axis = plt.subplots(1, 3, figsize=(16, 4.4))

    # A missing settled bar is informative: on the whole system no future case
    # clears the strict margin, so that test cannot be run there at all.
    honest = [
        ("all future cases", table["future_hmm_error_percent"], BLUE),
        ("the quarter watched longest",
         table["longest_follow_up_hmm_error_percent"], ORANGE),
        ("those meeting the strict margin",
         table["settled_hmm_error_percent"], GREEN),
        ("training cases (cannot fail)", table["hmm_error_percent"], GREY),
    ]
    bars(axis[0], x, honest, "error against measured energy (%)",
         "How far off on cases the chain never saw")
    label_bars(axis[0], x, honest)
    for position, row in enumerate(table.itertuples()):
        if pd.isna(row.settled_hmm_error_percent):
            axis[0].text(position + 0.1, 0.4, "no eligible cases", rotation=90,
                         ha="center", va="bottom", fontsize=7, color="dimgray")

    energy = [
        ("measured directly", table["measured_mean_case_energy"], GREY),
        ("predicted, HMM costs", table["case_energy_variants_hmm"], BLUE),
        ("predicted, OLS costs", table["case_energy_variants_ols"], GREEN),
    ]
    bars(axis[1], x, energy, "energy per complete case",
         "Energy of one complete case, training cases")

    chains = [
        ("variant chains", table["events_per_case_variants"], BLUE),
        ("pooled chain", table["events_per_case_pooled"], GREEN),
    ]
    bars(axis[2], x, chains, "expected events per case",
         "Consistency check: the two chains must agree")

    for panel in axis:
        panel.set_xticks(x)
        panel.set_xticklabels(scopes)

    figure.suptitle("Markov reward layer: energy of a whole case, "
                    "from the model's own transitions", fontsize=11)
    figure.tight_layout()
    path = OUT_DIR / "reward_layer.png"
    figure.savefig(path, dpi=140)
    print(f"saved -> {path.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
