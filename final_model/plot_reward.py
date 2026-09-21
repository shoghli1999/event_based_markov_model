"""
plot_reward.py
--------------
One figure for the Markov reward layer, read from reward_layer.csv.

It shows only the test that can fail: how far the predicted energy of one
complete case lies from the energy measured for cases the chain never saw.
Training cases are left out, because a chain built by counting reproduces the
cases it counted, and the agreement of the variant and pooled chains is left
out for the same reason; both are reported in the text as checks of the code.

The value plotted is the prediction minus the measured energy, as a share of the
measured energy, so a positive bar means the chain predicts too much. The groups
are all future cases, the quarter of them watched longest, and the ones that
began at least a full observation margin before the recording ends. On the whole
system no future case meets that margin, and the figure says so instead of
leaving a silent gap.

Output
------
    results_event_state/reward_layer.png
    images/reward_layer.png

Usage
-----
    python final_model/plot_reward.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results_event_state"
SCOPES = [("top3", "top three"), ("learnable", "whole system")]
GROUPS = [
    ("All future cases", "future_measured_mean_case_energy", "#4477AA"),
    ("Quarter watched longest", "longest_follow_up_measured_mean_case_energy", "#EE7733"),
    ("Strict observation margin", "settled_measured_mean_case_energy", "#228833"),
]


def main():
    """Draw the reward-layer figure from the saved table."""
    table = pd.read_csv(OUT_DIR / "reward_layer.csv").set_index("scope")
    figure, axis = plt.subplots(figsize=(5.58, 3.8))
    width = 0.26

    for index, (name, column, colour) in enumerate(GROUPS):
        for position, (scope, _) in enumerate(SCOPES):
            predicted = table.loc[scope, "case_energy_variants_hmm"]
            measured = table.loc[scope, column]
            x = position + (index - (len(GROUPS) - 1) / 2) * width
            if np.isnan(measured):
                axis.text(x, 0.4, "none\nqualify", ha="center", va="bottom",
                          fontsize=10, color="dimgray")
                continue
            value = 100.0 * (predicted - measured) / measured
            axis.bar(x, value, width, color=colour, label=name if position == 0 else None)
            axis.text(x, value + (0.3 if value >= 0 else -0.3), f"{value:+.2f}",
                      ha="center", va="bottom" if value >= 0 else "top", fontsize=10)

    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xticks(range(len(SCOPES)), [label for _, label in SCOPES])
    axis.set_xlim(-0.5, len(SCOPES) - 0.5)
    axis.set_ylim(-3, 21)
    axis.set_ylabel("Prediction above measured energy (%)")
    axis.set_title("How far the predicted energy of a case\n"
                   "lies from the measured one", fontsize=11)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=10, loc="upper left")
    figure.tight_layout()
    for folder in [OUT_DIR, ROOT / "images"]:
        path = folder / "reward_layer.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    plt.close(figure)


if __name__ == "__main__":
    main()
