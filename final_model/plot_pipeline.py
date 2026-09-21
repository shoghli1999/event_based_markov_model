"""
plot_pipeline.py
----------------
The pipeline figure for the start of the methodology chapter.

Drawn from code rather than by hand so it cannot drift away from the steps the
scripts actually take. The bottom row holds the four things that are measured:
the activity costs, the reconstruction of the held-out meter, the attribution of
hidden events, and the energy of a complete case.

The figure is drawn at the width it is printed at, so that its labels stay as
readable on paper as the text around them; what each box does in detail belongs
to the caption and to the sections that follow it.

Output
------
    images/pipeline.png
    results_event_state/pipeline.png

Usage
-----
    python final_model/plot_pipeline.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
REAL, MADE, MODEL, OUT = "#4477AA", "#EE7733", "#228833", "#AA3377"

# title, column, row, colour
BOXES = [
    ("BPI Challenge\n2019 log", 0, 6, REAL),
    ("Synthetic\nenergy", 1.5, 6, MADE),
    ("Building\nbackground", 3, 6, MADE),
    ("Variants and\nscopes", 0, 5, REAL),
    ("Interval\naggregation", 1.5, 4, MODEL),
    ("Chronological\nsplit", 1.5, 3, MODEL),
    ("Cost\nestimation", 0.5, 2, MODEL),
    ("Event-state\nHMM", 2.5, 2, MODEL),
    ("Decoding", 1.5, 1, MODEL),
    ("Markov reward\nlayer", 3, 1, MODEL),
    ("Activity\ncosts", 0, 0, OUT),
    ("Rebuilding\nthe meter", 1, 0, OUT),
    ("Attribution", 2, 0, OUT),
    ("Energy of a\ncomplete case", 3, 0, OUT),
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

WIDTH, HEIGHT, GAP_X, GAP_Y = 1.42, 0.78, 1.55, 1.2
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
    lean = 0.0 if abs(x0 - x1) < 0.01 else (0.35 if x1 > x0 else -0.35)
    return (x0 + lean, y0 - HEIGHT / 2), (x1 - lean, y1 + HEIGHT / 2)


def main():
    """Draw the pipeline figure and save it into images/."""
    figure, axis = plt.subplots(figsize=(5.58, 6.05))
    for title, column, row, colour in BOXES:
        x, y = place(column, row)
        axis.add_patch(FancyBboxPatch(
            (x - WIDTH / 2, y - HEIGHT / 2), WIDTH, HEIGHT,
            boxstyle="round,pad=0.03", linewidth=1.5,
            edgecolor=colour, facecolor=colour + "18", zorder=3))
        axis.text(x, y, title, ha="center", va="center", zorder=4,
                  fontsize=10, fontweight="bold", color=colour, linespacing=1.25)

    for source, target in ARROWS:
        start, end = edges(BOXES[source][1:3], BOXES[target][1:3])
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11, zorder=2,
            linewidth=1.1, color="#777777", shrinkA=0, shrinkB=0))

    top = ROWS * GAP_Y + HEIGHT + 0.35
    for index, (colour, label) in enumerate([(REAL, "from the event log"),
                                             (MADE, "generated"),
                                             (MODEL, "processing and model"),
                                             (OUT, "what is measured")]):
        y = top + 0.55 * (1 - index // 2)
        x = 0.1 + 3.1 * (index % 2)
        axis.add_patch(FancyBboxPatch(
            (x, y), 0.22, 0.2, boxstyle="round,pad=0.02", linewidth=1.4,
            edgecolor=colour, facecolor=colour + "18"))
        axis.text(x + 0.35, y + 0.1, label, fontsize=10, va="center", color="#333333")

    axis.set_xlim(-0.2, 3 * GAP_X + WIDTH + 0.2)
    axis.set_ylim(-0.3, top + 1.3)
    axis.axis("off")
    figure.tight_layout()
    for folder in ["images", "results_event_state"]:
        path = ROOT / folder / "pipeline.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    plt.close(figure)


if __name__ == "__main__":
    main()
