"""
plot_decoder_example.py
-----------------------
The four decoding methods side by side on one case of five events.

The three model-based decoders choose an activity for each held-out event in
turn, while the variant-first decoder chooses one complete training path for the
whole case. The sketch shows that difference, together with the evidence each
method reads. It carries no measured numbers; the scores of the four methods are
reported in the results chapter.

Output
------
    images/decoder_example.png
    results_event_state/decoder_example.png

Usage
-----
    python final_model/plot_decoder_example.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INK, GREY, BLUE, DARK, ORANGE = "#1f2a44", "#8a8a8a", "#4477AA", "#2b4f7d", "#EE7733"

SLOTS = 5
SLOT_W, SLOT_H, SLOT_GAP = 1.82, 1.05, 0.22

METHODS = [
    ("Position-only baseline", "case length and position only",
     "each event named on its own, without the meter", GREY),
    ("No-transition control", "energy and timing",
     "each event named on its own, from the meter and the timing", BLUE),
    ("Pooled decoder", "energy, timing and transitions",
     "built step by step from the transition table", DARK),
]
# The two most frequent variants of the log, which differ only in the order of the
# vendor invoice and the goods receipt.
PATHS = [
    (["Create\nPO item", "Vendor\ninvoice", "Goods\nreceipt", "Invoice\nreceipt", "Clear\ninvoice"],
     "most frequent training path"),
    (["Create\nPO item", "Goods\nreceipt", "Vendor\ninvoice", "Invoice\nreceipt", "Clear\ninvoice"],
     "second path, chosen here"),
]


def row_x(index):
    """Left edge of the slot at this position in the case."""
    return index * (SLOT_W + SLOT_GAP)


def slot(axis, x, y, colour, text="", filled=True):
    """One held-out event, drawn as a box that waits for an activity name."""
    axis.add_patch(FancyBboxPatch(
        (x, y - SLOT_H / 2), SLOT_W, SLOT_H,
        boxstyle="round,pad=0,rounding_size=0.08", linewidth=1.3,
        edgecolor=colour, facecolor=colour + "20" if filled else "white", zorder=3))
    if text:
        axis.text(x + SLOT_W / 2, y, text, ha="center", va="center", fontsize=10,
                  color=INK, zorder=5, linespacing=1.15)


def arrow(axis, start, end, colour):
    """A short arrow between two slots."""
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                   lw=1.1, color=colour, shrinkA=0, shrinkB=0, zorder=4))


def draw() -> plt.Figure:
    """Draw the four decoders, one block each, on the same five events."""
    right = row_x(SLOTS - 1) + SLOT_W
    figure, axis = plt.subplots(figsize=(5.58, 5.4))
    axis.set_xlim(-0.15, right + 0.15)
    axis.set_ylim(-4.3, 12.7)
    axis.axis("off")

    axis.text(0, 12.2, "one case of five held-out events, activity names hidden",
              fontsize=10, color="#333333")
    for index in range(SLOTS):
        axis.text(row_x(index) + SLOT_W / 2, 11.75, f"event {index + 1}", fontsize=10,
                  color="#555555", ha="center")

    top = 10.9
    for name, evidence, note, colour in METHODS:
        axis.text(0, top, name, fontsize=10.5, color=INK, fontweight="bold")
        axis.text(0, top - 0.5, evidence, fontsize=10, color="#555555")
        line = top - 1.45
        for index in range(SLOTS):
            slot(axis, row_x(index), line, colour, "?")
        if name.startswith("Pooled"):
            for index in range(SLOTS - 1):
                arrow(axis, (row_x(index) + SLOT_W, line), (row_x(index + 1), line), colour)
        axis.text(0, line - 0.95, note, fontsize=10, color="#555555")
        top -= 3.0

    colour = ORANGE
    axis.text(0, top, "Variant-first decoder", fontsize=10.5, color=INK, fontweight="bold")
    axis.text(0, top - 0.5, "energy, timing and path frequency", fontsize=10, color="#555555")
    line = top - 1.95
    for path, label in PATHS:
        chosen = "chosen" in label
        axis.text(0, line + 0.85, label, fontsize=10,
                  color=INK if chosen else "#777777",
                  fontweight="bold" if chosen else "normal")
        for index, activity in enumerate(path):
            slot(axis, row_x(index), line, colour if chosen else "#c3c3c3", activity, chosen)
        line -= 2.15
    axis.text(0, line + 0.55, "one whole path for the case, scored against the meter,\n"
              "the timing and how often the path occurred",
              fontsize=10, color="#555555", linespacing=1.25)

    figure.tight_layout()
    return figure


def main() -> None:
    """Draw the sketch and save it into images/ and results_event_state/."""
    figure = draw()
    for folder in ["images", "results_event_state"]:
        path = ROOT / folder / "decoder_example.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    plt.close(figure)


if __name__ == "__main__":
    main()
