"""
plot_state_example.py
---------------------
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
------
    results_event_state/state_example.csv
    results_event_state/state_example.png
    images/state_example.png

Usage
-----
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
    """Draw the states as a chain read from top to bottom, from the saved table."""
    p = {(row["from"], row["to"]): row["probability"] for _, row in table.iterrows()}
    ink, blue, grey = "#1f2a44", "#2b6cb0", "#4b5563"
    box_left, box_right, box_h, gap = 1.1, 7.6, 0.86, 0.85

    figure, axis = plt.subplots(figsize=(5.58, 4.95))
    axis.set_xlim(0, 10)
    top = 0.6 + (len(STATES) - 1) * (box_h + gap) + box_h
    axis.set_ylim(0.3, top + 1.5)
    axis.axis("off")

    def centre(index):
        return top - box_h / 2 - index * (box_h + gap)

    axis.plot([(box_left + box_right) / 2], [top + 1.05], "o", ms=9, color=ink)
    axis.text((box_left + box_right) / 2 + 0.25, top + 1.05, "start", fontsize=10,
              color=ink, va="center")
    axis.add_patch(FancyArrowPatch(((box_left + box_right) / 2, top + 0.92),
                                   ((box_left + box_right) / 2, centre(0) + box_h / 2 + 0.06),
                                   arrowstyle="-|>", mutation_scale=14, lw=1.5, color=grey,
                                   shrinkA=0, shrinkB=0))
    axis.text((box_left + box_right) / 2 + 0.25, top + 0.45,
              f"{p[('start', STATES[0])]:.3f}", fontsize=10, color=ink, va="center")

    for index, state in enumerate(STATES):
        y = centre(index)
        axis.add_patch(FancyBboxPatch((box_left, y - box_h / 2), box_right - box_left, box_h,
                                      boxstyle="round,pad=0,rounding_size=0.12",
                                      facecolor="#eef4fb", edgecolor=blue, lw=1.6, zorder=3))
        axis.text((box_left + box_right) / 2, y, state, ha="center", va="center",
                  fontsize=10.5, color=ink, zorder=5)
        if index < len(STATES) - 1:
            below = centre(index + 1)
            x = (box_left + box_right) / 2
            axis.add_patch(FancyArrowPatch((x, y - box_h / 2 - 0.06), (x, below + box_h / 2 + 0.06),
                                           arrowstyle="-|>", mutation_scale=14, lw=1.5,
                                           color=grey, shrinkA=0, shrinkB=0, zorder=4))
            axis.text(x + 0.25, (y + below) / 2, f"{p[(state, STATES[index + 1])]:.3f}",
                      fontsize=10, color=ink, va="center")

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
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    print(f"saved -> {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
