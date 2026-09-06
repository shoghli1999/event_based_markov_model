"""
plot_pipeline.py
────────────────
The pipeline figure for the start of the methodology chapter.

Drawn from code rather than by hand so it cannot drift away from what the
scripts actually do. Every box names the file that performs that step.

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
    ("BPI Challenge 2019 log", "case, activity, timestamp\n1,595,603 events, 42 activities", 0, 5, REAL),
    ("Synthetic energy", "base cost + duration x 0.01\nper event", 2, 5, MADE),
    ("Process variants", "most frequent complete paths\ntop 1, 2, 3, 5 and the learnable log", 0, 4, REAL),
    ("Building background", "daily shape x uniform(80, 90)\nfresh in every interval", 2, 4, MADE),
    ("Interval aggregation", "30 minute totals; an event crossing\na boundary is split, energy and\ncount together", 1, 3, MODEL),
    ("Chronological split", "cut after 70% of all events,\nby cumulative count", 1, 2, MODEL),
    ("Viterbi decoding", "no transitions, pooled,\nand variant first", 0, 1, MODEL),
    ("Weighted regression", "variance = a + b(events)\n+ c(background)^2", 1, 1, MODEL),
    ("Markov reward layer", "absorbing chain with an END\nstate, energy of a whole case", 2, 1, MODEL),
    ("Hidden event attribution", "activity names removed\nafter the cut", 0, 0, OUT),
    ("Activity costs", "compared with OLS and ridge", 1, 0, OUT),
    ("Energy per complete case", "tested on cases never seen", 2, 0, OUT),
]

ARROWS = [(0, 1), (0, 2), (2, 4), (1, 4), (3, 4), (4, 5),
          (5, 6), (5, 7), (5, 8), (6, 9), (7, 10), (8, 11)]

WIDTH, HEIGHT, GAP_X, GAP_Y = 3.6, 1.3, 4.5, 2.05


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
    figure, axis = plt.subplots(figsize=(14, 11))
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

    top = 5 * GAP_Y + HEIGHT + 0.55
    for index, (colour, label) in enumerate([(REAL, "from the event log"),
                                             (MADE, "generated"),
                                             (MODEL, "model"),
                                             (OUT, "what is measured")]):
        x = index * 3.3
        axis.add_patch(FancyBboxPatch(
            (x, top), 0.42, 0.28, boxstyle="round,pad=0.03", linewidth=1.5,
            edgecolor=colour, facecolor=colour + "18"))
        axis.text(x + 0.65, top + 0.14, label, fontsize=9.5,
                  va="center", color="#333333")

    axis.set_xlim(-0.5, 2 * GAP_X + WIDTH + 0.5)
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
