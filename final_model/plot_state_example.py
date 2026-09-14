"""
plot_state_example.py
─────────────────────
A few states of the pooled whole-system model, drawn from the model itself.

One hidden state is one activity, so a picture of the model is a picture of the
process. This script counts the start and transition probabilities exactly as
event_state_hmm.py does for decoding, saves the ones it draws to a table, and
draws the figure from that table only.

Only selected states and transitions are shown. The pooled model has 35 states
and no end state, so the probabilities leaving a drawn state do not sum to one.
The strongest transition out of Clear Invoice is drawn so that the figure does
not suggest the process ends there.

Output
──────
    results_event_state/state_example.csv
    results_event_state/state_example.png
    images/state_example.png

Usage
─────
    python final_model/plot_state_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import event_state_hmm as model  # noqa: E402

STATES = [
    "Create Purchase Order Item",
    "Vendor creates invoice",
    "Record Goods Receipt",
    "Record Invoice Receipt",
    "Clear Invoice",
]
# (from, to) by position in STATES: the most frequent path, the two ways of
# skipping or reversing the goods receipt, and the way back from Clear Invoice.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3), (0, 2), (2, 1), (4, 3)]
TABLE = model.OUT_DIR / "state_example.csv"


def probability_table() -> pd.DataFrame:
    """Start and transition probabilities of the drawn states, from complete training cases."""
    problem = model.load_problem("learnable")
    start, transition = model._transition_parameters(problem)
    index = {activity: position for position, activity in enumerate(problem.activities)}
    rows = [{"from": "start", "to": STATES[0], "probability": float(start[index[STATES[0]]])}]
    for a, b in EDGES:
        rows.append({
            "from": STATES[a],
            "to": STATES[b],
            "probability": float(transition[index[STATES[a]], index[STATES[b]]]),
        })
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame) -> plt.Figure:
    """Draw the states left to right with their probabilities, read from the saved table."""
    p = {(row["from"], row["to"]): row["probability"] for _, row in table.iterrows()}
    x = {state: 3.2 * position for position, state in enumerate(STATES)}
    width, height, gap = 2.3, 1.0, 0.07
    ink, blue, grey = "#1f2a44", "#2b6cb0", "#4b5563"

    figure, axis = plt.subplots(figsize=(13.5, 4.8))
    axis.set_xlim(-2.3, 14.2)
    axis.set_ylim(-2.2, 2.3)
    axis.axis("off")
    axis.set_title("Selected states and transitions", fontsize=13, fontweight="bold",
                   color=ink, loc="left")

    for state in STATES:
        axis.add_patch(FancyBboxPatch(
            (x[state] - width / 2, -height / 2), width, height,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor="#eef4fb", edgecolor=blue, lw=1.8, zorder=3))
        words = state.split(" ")
        middle = (len(words) + 1) // 2 if len(words) > 2 else len(words)
        text = " ".join(words[:middle]) + ("\n" + " ".join(words[middle:]) if words[middle:] else "")
        axis.text(x[state], 0, text, ha="center", va="center", fontsize=10.5,
                  fontweight="bold", color=ink, zorder=5, linespacing=1.25)

    def label(px, py, value):
        axis.text(px, py, f"{value:.3f}", ha="center", va="center", fontsize=10, color=ink,
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"), zorder=6)

    def arrow(start, end, rad=0.0):
        axis.add_patch(FancyArrowPatch(
            start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=17, lw=1.7, color=grey, shrinkA=0, shrinkB=0, zorder=4))

    def straight(a, b):
        arrow((x[a] + width / 2 + gap, 0), (x[b] - width / 2 - gap, 0))
        label((x[a] + x[b]) / 2, 0.24, p[(a, b)])

    def arc(a, xa, b, xb, rad, above):
        y = (height / 2 + gap) if above else -(height / 2 + gap)
        arrow((xa, y), (xb, y), rad)
        apex = abs(rad) * abs(xb - xa) / 2
        label((xa + xb) / 2, y + (apex if above else -apex), p[(a, b)])

    c, v, r, i, l = STATES
    for a, b in [(c, v), (v, r), (r, i), (i, l)]:
        straight(a, b)
    arc(v, x[v], i, x[i], rad=-0.24, above=True)
    arc(c, x[c], r, x[r] + 0.35, rad=0.34, above=False)
    arc(r, x[r] - 0.55, v, x[v] + 0.55, rad=-0.30, above=False)
    arc(l, x[l] - 0.55, i, x[i] + 0.55, rad=-0.30, above=False)

    axis.plot([-1.95], [0], "o", ms=11, color=ink, zorder=5)
    arrow((-1.82, 0), (x[c] - width / 2 - gap, 0))
    axis.text(-1.95, 0.34, "start", ha="center", fontsize=10, color=ink)
    label((-1.82 + x[c] - width / 2) / 2, 0.24, p[("start", c)])
    figure.tight_layout()
    return figure


def main() -> None:
    """Count the probabilities, save them, and draw the figure from the saved table."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    probability_table().to_csv(TABLE, index=False)
    figure = draw(pd.read_csv(TABLE))
    for folder in ["images", "results_event_state"]:
        path = ROOT / folder / "state_example.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    print(f"saved -> {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
