"""
plot_end_example.py
-------------------
The states of plot_state_example.py again, now in the absorbing chain of the
reward layer.

The reward layer adds an END state: every completed training case takes one
final step from its last activity into END. This script counts the start, the
steps along the most frequent path and the steps into END exactly as
reward_layer.py does, saves them to a table, and draws the figure from that
table only.

Because each row now also leads to END, the steps between activities are
slightly lower than in the decoding model wherever a case can end. Transitions
to other activities are not drawn, so the arrows leaving a state do not sum to
one.

Output
------
    results_event_state/end_example.csv
    results_event_state/end_example.png
    images/end_example.png

Usage
-----
    python final_model/plot_end_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "final_model"))

import event_state_hmm as model  # noqa: E402
from reward_layer import SMOOTHING, training_cases  # noqa: E402

STATES = [
    "Create Purchase Order Item",
    "Vendor creates invoice",
    "Record Goods Receipt",
    "Record Invoice Receipt",
    "Clear Invoice",
]
TABLE = model.OUT_DIR / "end_example.csv"


def probability_table() -> pd.DataFrame:
    """Start, path and END probabilities of the drawn states, counted as in reward_layer.py."""
    problem = model.load_problem("learnable")
    K = len(problem.activities)
    start = np.full(K, SMOOTHING)
    trans = np.full((K, K + 1), SMOOTHING)          # last column is END
    for _, case in training_cases(problem).groupby("case", sort=False):
        states = case["state"].to_numpy(int)
        start[states[0]] += 1.0
        np.add.at(trans, (states[:-1], states[1:]), 1.0)
        trans[states[-1], K] += 1.0
    start /= start.sum()
    trans /= trans.sum(axis=1, keepdims=True)

    index = {activity: position for position, activity in enumerate(problem.activities)}
    rows = [{"from": "start", "to": STATES[0], "probability": float(start[index[STATES[0]]])}]
    for a, b in zip(STATES[:-1], STATES[1:]):
        rows.append({"from": a, "to": b, "probability": float(trans[index[a], index[b]])})
    for state in STATES:
        rows.append({"from": state, "to": "END", "probability": float(trans[index[state], K])})
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame) -> plt.Figure:
    """Draw the chain from top to bottom with the END state below it."""
    p = {(row["from"], row["to"]): row["probability"] for _, row in table.iterrows()}
    ink, blue, grey, red = "#1f2a44", "#2b6cb0", "#4b5563", "#9b2c2c"
    box_left, box_right, box_h, gap = 0.6, 6.4, 0.86, 0.9
    middle = (box_left + box_right) / 2

    figure, axis = plt.subplots(figsize=(5.58, 6.1))
    axis.set_xlim(0, 10)
    top = 1.6 + (len(STATES) - 1) * (box_h + gap) + box_h
    axis.set_ylim(-0.2, top + 1.5)
    axis.axis("off")

    def centre(index):
        return top - box_h / 2 - index * (box_h + gap)

    axis.plot([middle], [top + 1.05], "o", ms=9, color=ink)
    axis.text(middle + 0.25, top + 1.05, "start", fontsize=10, color=ink, va="center")
    axis.add_patch(FancyArrowPatch((middle, top + 0.92), (middle, centre(0) + box_h / 2 + 0.06),
                                   arrowstyle="-|>", mutation_scale=14, lw=1.5, color=grey,
                                   shrinkA=0, shrinkB=0))
    axis.text(middle + 0.25, top + 0.45, f"{p[('start', STATES[0])]:.3f}", fontsize=10,
              color=ink, va="center")

    end_y = centre(len(STATES) - 1) - box_h - 0.75
    for index, state in enumerate(STATES):
        y = centre(index)
        axis.add_patch(FancyBboxPatch((box_left, y - box_h / 2), box_right - box_left, box_h,
                                      boxstyle="round,pad=0,rounding_size=0.12",
                                      facecolor="#eef4fb", edgecolor=blue, lw=1.6, zorder=3))
        axis.text(middle, y, state, ha="center", va="center", fontsize=10.5, color=ink, zorder=5)
        if index < len(STATES) - 1:
            below = centre(index + 1)
            axis.add_patch(FancyArrowPatch((middle, y - box_h / 2 - 0.06),
                                           (middle, below + box_h / 2 + 0.06),
                                           arrowstyle="-|>", mutation_scale=14, lw=1.5,
                                           color=grey, shrinkA=0, shrinkB=0, zorder=4))
            axis.text(middle + 0.25, (y + below) / 2,
                      f"{p[(state, STATES[index + 1])]:.3f}", fontsize=10, color=ink, va="center")
        # every activity can also end the case
        value = p[(state, "END")]
        text = f"{value:.3f}" if value >= 0.0005 else "<0.001"
        if index < len(STATES) - 1:
            axis.add_patch(FancyArrowPatch((box_right + 0.05, y), (box_right - 1.4, end_y + 0.5),
                                           arrowstyle="-|>", mutation_scale=11, lw=1.2, color=red,
                                           shrinkA=0, shrinkB=0, zorder=2,
                                           connectionstyle="arc3,rad=-0.35"))
            axis.text(box_right + 0.25, y, text, fontsize=10, color=red, va="center",
                      zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        else:
            axis.text(middle + 0.25, (y + end_y) / 2, text, fontsize=10, color=red,
                      va="center", zorder=6,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

    axis.add_patch(FancyBboxPatch((box_left + 1.4, end_y - box_h / 2), 3.0, box_h,
                                  boxstyle="round,pad=0,rounding_size=0.12",
                                  facecolor="#fbeeee", edgecolor=red, lw=1.6, zorder=3))
    axis.text(box_left + 2.9, end_y, "END", ha="center", va="center", fontsize=11,
              fontweight="bold", color=ink, zorder=5)
    axis.add_patch(FancyArrowPatch((middle, centre(len(STATES) - 1) - box_h / 2 - 0.06),
                                   (middle, end_y + box_h / 2 + 0.06), arrowstyle="-|>",
                                   mutation_scale=14, lw=1.5, color=red, shrinkA=0, shrinkB=0))
    figure.tight_layout()
    return figure


def main() -> None:
    """Count the probabilities, save them, and draw the figure from the saved table."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    probability_table().to_csv(TABLE, index=False)
    figure = draw(pd.read_csv(TABLE))
    for folder in ["images", "results_event_state"]:
        path = ROOT / folder / "end_example.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    print(f"saved -> {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
