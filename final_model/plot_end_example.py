"""
plot_end_example.py
───────────────────
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
──────
    results_event_state/end_example.csv
    results_event_state/end_example.png
    images/end_example.png

Usage
─────
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
    """Draw the states left to right and END below them, read from the saved table."""
    p = {(row["from"], row["to"]): row["probability"] for _, row in table.iterrows()}
    x = {state: 3.2 * position for position, state in enumerate(STATES)}
    width, height, gap = 2.3, 1.0, 0.07
    top, bottom = 1.0, -1.6
    ink, blue, grey, red = "#1f2a44", "#2b6cb0", "#4b5563", "#9b2c2c"

    figure, axis = plt.subplots(figsize=(13.5, 4.8))
    axis.set_xlim(-2.3, 14.2)
    axis.set_ylim(-2.3, 2.1)
    axis.axis("off")
    axis.set_title("Selected states with the END state", fontsize=13, fontweight="bold",
                   color=ink, loc="left")

    def box(cx, cy, box_width, text, face, edge):
        axis.add_patch(FancyBboxPatch(
            (cx - box_width / 2, cy - height / 2), box_width, height,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=face, edgecolor=edge, lw=1.8, zorder=3))
        axis.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                  fontweight="bold", color=ink, zorder=5, linespacing=1.25)

    def label(px, py, value):
        # A step into END that almost never happens would round to 0.000.
        text = f"{value:.3f}" if value >= 0.0005 else "<0.001"
        axis.text(px, py, text, ha="center", va="center", fontsize=10, color=ink,
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"), zorder=6)

    def arrow(start, end, color=grey):
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=17, lw=1.7, color=color,
            shrinkA=0, shrinkB=0, zorder=4))

    for state in STATES:
        words = state.split(" ")
        middle = (len(words) + 1) // 2 if len(words) > 2 else len(words)
        text = " ".join(words[:middle]) + ("\n" + " ".join(words[middle:]) if words[middle:] else "")
        box(x[state], top, width, text, "#eef4fb", blue)

    for a, b in zip(STATES[:-1], STATES[1:]):
        arrow((x[a] + width / 2 + gap, top), (x[b] - width / 2 - gap, top))
        label((x[a] + x[b]) / 2, top + 0.24, p[(a, b)])

    box(x[STATES[2]], bottom, x[STATES[-1]] - x[STATES[0]] + width, "END", "#fbeeee", red)
    for state in STATES:
        arrow((x[state], top - height / 2 - gap), (x[state], bottom + height / 2 + gap), red)
        label(x[state], (top + bottom) / 2, p[(state, "END")])

    first = STATES[0]
    axis.plot([-1.95], [top], "o", ms=11, color=ink, zorder=5)
    arrow((-1.82, top), (x[first] - width / 2 - gap, top))
    axis.text(-1.95, top + 0.34, "start", ha="center", fontsize=10, color=ink)
    label((-1.82 + x[first] - width / 2) / 2, top + 0.24, p[("start", first)])
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
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    print(f"saved -> {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
