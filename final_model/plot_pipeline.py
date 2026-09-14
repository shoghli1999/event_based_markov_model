"""
plot_pipeline.py
────────────────
The pipeline figure for the start of the methodology chapter.

Drawn from code rather than by hand so it cannot drift away from the steps the
scripts actually take. The bottom row holds the four things that are measured:
the activity costs, the reconstruction of the held-out meter, the attribution of
hidden events, and the energy of a complete case.

Output
──────
    images/pipeline.png
    results_event_state/pipeline.png

Usage
─────
    python final_model/plot_pipeline.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
REAL, MADE, MODEL, OUT = "#4477AA", "#EE7733", "#228833", "#AA3377"

# title, subtitle, column, row, colour
BOXES = [
    ("BPI Challenge 2019 log", "case, activity, timestamp\n1,595,603 events, 42 activities", 0, 6, REAL),
    ("Synthetic energy", "base cost + 0.01 x duration\nfor every event", 1.5, 6, MADE),
    ("Building background", "daily shape x uniform(80, 90),\nfresh in every interval", 3, 6, MADE),
    ("Process variants and scopes", "top 1, 2, 3 and 5 variants and\nthe whole system of 35 activities", 0, 5, REAL),
    ("Interval aggregation", "30-minute totals and counts, expected\nbackground removed; a crossing event\nis split, energy and count together", 1.5, 4, MODEL),
    ("Chronological split", "cut after 70% of all events;\ntraining part fits the models,\nheld-out part is decoded and scored", 1.5, 3, MODEL),
    ("Cost estimation", "weighted estimator,\nwith OLS and ridge as baselines", 0.5, 2, MODEL),
    ("Event-state HMM", "one state per activity: start,\ntransitions, energy and timing", 2.5, 2, MODEL),
    ("Decoding", "no-transition, pooled and variant-first,\nagainst a position-only baseline", 1.5, 1, MODEL),
    ("Markov reward layer", "absorbing chain\nwith an END state", 3, 1, MODEL),
    ("Activity costs", "error against\nthe true costs", 0, 0, OUT),
    ("Reconstruction", "held-out meter rebuilt from\nknown or decoded activities", 1, 0, OUT),
    ("Attribution", "activity and energy of\neach held-out event", 2, 0, OUT),
    ("Energy of a complete case", "tested on cases\nnever seen", 3, 0, OUT),
]

# (source, target) by position in BOXES
ARROWS = [
    (0, 1), (0, 3),                 # log to its energy and to the scopes
    (1, 4), (2, 4), (3, 4),         # energy, background and scope into the intervals
    (4, 5),                         # intervals and their counts into the split
    (5, 6), (5, 7),                 # training part into costs and the HMM
    (6, 7),                         # costs become the energy emissions
    (7, 8), (7, 9),                 # the HMM drives decoding and the reward layer
    (6, 10), (6, 11),               # costs, and the meter rebuilt from known activities
    (8, 11), (8, 12),               # the meter rebuilt from decoded activities, attribution
    (9, 13),                        # energy of a complete case
]

WIDTH, HEIGHT, GAP_X, GAP_Y = 3.6, 1.3, 4.2, 2.05
ROWS = max(row for *_, row, _ in BOXES)


def place(column, row):
    """Centre of the box in figure coordinates."""
    return column * GAP_X + WIDTH / 2, row * GAP_Y + HEIGHT / 2


def edges(source, target):
    """Where an arrow should leave one box and enter the other."""
    x0, y0 = place(*source)
    x1, y1 = place(*target)
    if abs(y0 - y1) < 0.01:
        side = WIDTH / 2 if x0 < x1 else -WIDTH / 2
        return (x0 + side, y0), (x1 - side, y1)
    # leave the bottom and enter the top, nudged sideways when the columns differ
    lean = 0.0 if abs(x0 - x1) < 0.01 else (0.9 if x1 > x0 else -0.9)
    return (x0 + lean, y0 - HEIGHT / 2), (x1 - lean, y1 + HEIGHT / 2)


def main():
    """Draw the pipeline figure and save it into images/."""
    figure, axis = plt.subplots(figsize=(15, 12.5))
    for title, subtitle, column, row, colour in BOXES:
        x, y = place(column, row)
        axis.add_patch(FancyBboxPatch(
            (x - WIDTH / 2, y - HEIGHT / 2), WIDTH, HEIGHT,
            boxstyle="round,pad=0.07", linewidth=1.7,
            edgecolor=colour, facecolor=colour + "18", zorder=3))
        axis.text(x, y + 0.34, title, ha="center", va="center", zorder=4,
                  fontsize=11.5, fontweight="bold", color=colour)
        axis.text(x, y - 0.20, subtitle, ha="center", va="center", zorder=4,
                  fontsize=8.6, color="#333333", linespacing=1.45)

    for source, target in ARROWS:
        start, end = edges(BOXES[source][2:4], BOXES[target][2:4])
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, zorder=2,
            linewidth=1.3, color="#777777", shrinkA=0, shrinkB=0))

    top = ROWS * GAP_Y + HEIGHT + 0.55
    for index, (colour, label) in enumerate([(REAL, "from the event log"),
                                             (MADE, "generated"),
                                             (MODEL, "processing and model"),
                                             (OUT, "what is measured")]):
        x = index * GAP_X
        axis.add_patch(FancyBboxPatch(
            (x, top), 0.42, 0.28, boxstyle="round,pad=0.03", linewidth=1.5,
            edgecolor=colour, facecolor=colour + "18"))
        axis.text(x + 0.65, top + 0.14, label, fontsize=9.5,
                  va="center", color="#333333")

    axis.set_xlim(-0.5, 3 * GAP_X + WIDTH + 0.5)
    axis.set_ylim(-0.6, top + 0.9)
    axis.axis("off")
    figure.tight_layout()
    for folder in ["images", "results_event_state"]:
        path = ROOT / folder / "pipeline.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    plt.close(figure)


if __name__ == "__main__":
    main()
