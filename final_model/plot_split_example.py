"""
plot_split_example.py
---------------------
The chronological split, and how a case is used on either side of it.

The cut falls on the half-hour boundary by which about seventy percent of the
events have happened. A case is used according to where it lies relative to that
boundary, and this figure draws the three possibilities on one timeline. The
cut, the end of the recording and the number of cases in each group are counted
from the whole-system scope rather than written by hand, so the figure cannot
drift away from the experiment.

Output
------
    results_event_state/split_example.csv
    results_event_state/split_example.png
    images/split_example.png

Usage
-----
    python final_model/plot_split_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "final_model"))

import event_state_hmm as model  # noqa: E402
from reward_layer import recording_end_timestamp  # noqa: E402

TABLE = model.OUT_DIR / "split_example.csv"
INK, TRAIN, SPAN, TEST, CUT = "#1f2a44", "#4477AA", "#7d6f9c", "#EE7733", "#9b2c2c"


def split_facts() -> pd.DataFrame:
    """Cut time, recording end and the size of the three case groups."""
    problem = model.load_problem("learnable")
    events = problem.events
    first = events.groupby("case", sort=False)["bin"].min()
    last = events.groupby("case", sort=False)["bin"].max()
    start = events["ts"].dt.floor(model.FREQ).min()
    test = events["bin"].to_numpy(int) >= problem.cut
    rows = [
        {"fact": "timeline_start", "value": str(start)},
        {"fact": "cut_time", "value": str(start + problem.cut * pd.Timedelta(model.FREQ))},
        {"fact": "recording_end", "value": str(recording_end_timestamp())},
        {"fact": "last_event", "value": str(events["ts"].max())},
        {"fact": "training_event_percent", "value": 100.0 * float((~test).mean())},
        {"fact": "cases_completed_before_cut", "value": int((last < problem.cut).sum())},
        {"fact": "cases_spanning_cut",
         "value": int(((first < problem.cut) & (last >= problem.cut)).sum())},
        {"fact": "cases_after_cut", "value": int((first >= problem.cut).sum())},
    ]
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame) -> plt.Figure:
    """Draw the timeline, the cut and one case of each kind, from the saved table."""
    value = dict(zip(table["fact"], table["value"]))
    start = pd.Timestamp(value["timeline_start"])
    cut = pd.Timestamp(value["cut_time"])
    end = pd.Timestamp(value["recording_end"])
    share = float(value["training_event_percent"])
    counts = {name: f"{int(float(value[name])):,}" for name in
              ["cases_completed_before_cut", "cases_spanning_cut", "cases_after_cut"]}

    figure, axis = plt.subplots(figsize=(5.58, 4.15))
    left, right = mdates.date2num(start), mdates.date2num(end)
    middle = mdates.date2num(cut)
    axis.set_xlim(left - 10, right + 10)
    axis.set_ylim(0.0, 4.0)
    axis.get_yaxis().set_visible(False)
    for side in ["left", "right", "top"]:
        axis.spines[side].set_visible(False)

    axis.axvspan(left, middle, color=TRAIN, alpha=0.09)
    axis.axvspan(middle, right, color=TEST, alpha=0.09)
    axis.axvline(middle, color=CUT, lw=1.8)
    axis.text(middle - 8, 3.86, f"cut after {share:.2f}% of the events ", color=CUT,
              fontsize=10, ha="right", va="center")
    axis.text((left + middle) / 2, 3.45, "training", color=TRAIN, fontsize=11,
              ha="center", fontweight="bold")
    axis.text((middle + right) / 2, 3.45, "test", color=TEST, fontsize=11,
              ha="center", fontweight="bold")

    labels = []

    def case(y, begins, finishes, colour, title, note):
        x0, x1 = mdates.date2num(begins), mdates.date2num(finishes)
        axis.add_patch(Rectangle((x0, y - 0.08), x1 - x0, 0.16, facecolor=colour,
                                 edgecolor=colour, zorder=3))
        # The title and the note belong to the bar below and above them, so they
        # start where the bar starts, and swing to its right end when a line
        # would otherwise run past the timeline.
        labels.append((axis.text(x0, y + 0.24, title, fontsize=10, color=INK,
                                 fontweight="bold"), x1))
        labels.append((axis.text(x0, y - 0.46, note, fontsize=10, color="#333333",
                                 linespacing=1.25), x1))

    day = pd.Timedelta(days=1)
    case(2.85, start + 20 * day, cut - 55 * day, TRAIN,
         f"completed before the cut, {counts['cases_completed_before_cut']} cases",
         "transition and timing parameters,\ntraining paths, reward chain")
    case(1.75, cut - 70 * day, cut + 45 * day, SPAN,
         f"spanning the cut, {counts['cases_spanning_cut']} cases",
         "earlier events in the costs,\nlater events decoded")
    case(0.65, cut + 20 * day, end - 12 * day, TEST,
         f"beginning after the cut, {counts['cases_after_cut']} cases",
         "decoded, and used for the\nreward-layer test")

    figure.canvas.draw()
    inside = axis.get_window_extent()
    for label, end_of_bar in labels:
        if label.get_window_extent().x1 > inside.x1:
            label.set_x(end_of_bar)
            label.set_horizontalalignment("right")

    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.tick_params(axis="x", labelsize=10, colors="#333333")
    figure.tight_layout()
    return figure


def main() -> None:
    """Count the facts, save them, and draw the figure from the saved table."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_facts().to_csv(TABLE, index=False)
    figure = draw(pd.read_csv(TABLE))
    for folder in ["images", "results_event_state"]:
        path = ROOT / folder / "split_example.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    print(f"saved -> {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
