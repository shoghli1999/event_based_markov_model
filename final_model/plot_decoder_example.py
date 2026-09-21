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
SLOT_W, SLOT_H, SLOT_GAP = 1.7, 0.66, 0.26
LEFT = 6.4

METHODS = [
    ("Position-only baseline", "case length and position, no meter", GREY),
    ("No-transition control", "energy and timing", BLUE),
    ("Pooled decoder", "energy, timing and transitions", DARK),
    ("Variant-first decoder", "energy, timing and path frequency", ORANGE),
]
# The two most frequent variants of the log, which differ only in the order of the
# vendor invoice and the goods receipt.
PATHS = [
    (["Create\nPO item", "Vendor\ninvoice", "Goods\nreceipt", "Invoice\nreceipt", "Clear\ninvoice"],
     "most frequent training path"),
    (["Create\nPO item", "Goods\nreceipt", "Vendor\ninvoice", "Invoice\nreceipt", "Clear\ninvoice"],
     "second path, chosen here"),
]


def slot(axis, x, y, colour, text="", filled=True):
    """One held-out event, drawn as a box that waits for an activity name."""
    axis.add_patch(FancyBboxPatch(
        (x, y - SLOT_H / 2), SLOT_W, SLOT_H,
        boxstyle="round,pad=0,rounding_size=0.08", linewidth=1.4,
        edgecolor=colour, facecolor=colour + "20" if filled else "white", zorder=3))
    if text:
        axis.text(x + SLOT_W / 2, y, text, ha="center", va="center", fontsize=7.6,
                  color=INK, zorder=5, linespacing=1.2)


def row_x(index):
    """Left edge of the slot at this position in the case."""
    return LEFT + index * (SLOT_W + SLOT_GAP)


def arrow(axis, start, end, colour):
    """A short arrow between two slots."""
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                   lw=1.2, color=colour, shrinkA=0, shrinkB=0, zorder=4))


def draw() -> plt.Figure:
    """Draw the four decoders, one per block, on the same five events."""
    figure, axis = plt.subplots(figsize=(12.4, 7.0))
    axis.set_xlim(0, row_x(SLOTS - 1) + SLOT_W + 3.6)
    axis.set_ylim(-2.15, 5.2)
    axis.axis("off")

    axis.text(LEFT, 4.85, "the five held-out events of one case, activity names hidden",
              fontsize=10, color="#333333")
    for index in range(SLOTS):
        axis.text(row_x(index) + SLOT_W / 2, 4.45, f"event {index + 1}", fontsize=9,
                  color="#555555", ha="center")

    rows = [3.75, 2.55, 1.35]
    notes = ["each event named on its own, without reading the meter",
             "each event named on its own, from the meter and the timing",
             "one step at a time, with the transition table"]
    for (name, evidence, colour), y, note in zip(METHODS, rows, notes):
        axis.text(0, y + 0.18, name, fontsize=10.5, color=INK, fontweight="bold")
        axis.text(0, y - 0.24, evidence, fontsize=9, color="#555555")
        for index in range(SLOTS):
            slot(axis, row_x(index), y, colour, "?")
        if name.startswith("Pooled"):
            for index in range(SLOTS - 1):
                arrow(axis, (row_x(index) + SLOT_W, y), (row_x(index + 1), y), colour)
        axis.text(row_x(0), y - 0.58, note, fontsize=8.5, color="#555555")

    name, evidence, colour = METHODS[3]
    axis.text(0, 0.33, name, fontsize=10.5, color=INK, fontweight="bold")
    axis.text(0, -0.09, evidence, fontsize=9, color="#555555")
    for offset, (path, label) in enumerate(PATHS):
        y = 0.15 - offset * 1.0
        chosen = "chosen" in label
        for index, activity in enumerate(path):
            slot(axis, row_x(index), y, colour if chosen else "#c3c3c3", activity, chosen)
        axis.text(row_x(SLOTS - 1) + SLOT_W + 0.25, y, label, fontsize=9,
                  color=INK if chosen else "#777777", va="center",
                  fontweight="bold" if chosen else "normal")
    axis.text(row_x(0), -1.38, "and the further training paths of the same length, not drawn",
              fontsize=8.5, color="#777777")
    axis.text(row_x(0), -1.85,
              "one whole path for the case, scored against the meter, the timing and how "
              "often the path occurred", fontsize=8.5, color="#555555")

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
