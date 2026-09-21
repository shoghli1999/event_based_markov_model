"""
plot_signal_example.py
----------------------
Two days of the generated meter signal on the top three scope.

The text describes the signal through its formula: a daily shape multiplied by a
fresh random number in every interval, plus the energy of the events that happen
in it. This figure shows what that looks like over two days. The upper panel
draws the background alone, the predictable part against the part that is drawn
again in every interval, which is the noise every estimator has to live with and
which grows with the shape. The lower panel puts that background beside the
meter total on a logarithmic scale: the events dominate the busy hours by orders
of magnitude, while in a quiet interval the meter is background and nothing
else.

Output
------
    results_event_state/signal_example.csv
    results_event_state/signal_example.png
    images/signal_example.png

Usage
-----
    python final_model/plot_signal_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_pipeline as pipeline  # noqa: E402
import event_state_hmm as model  # noqa: E402

TABLE = model.OUT_DIR / "signal_example.csv"
WINDOW = (pd.Timestamp("2018-04-30 00:00"), pd.Timestamp("2018-05-02 00:00"))
METER, BACK, EVENT, NOISE = "#1f2a44", "#4477AA", "#EE7733", "#9b2c2c"


def window_signal() -> pd.DataFrame:
    """The generated series of the example window, as the pipeline builds it."""
    events = pipeline.select_coverage(model._simulated_complete_events(), "3")
    signal, _ = pipeline.build_signal_and_matrix(events)
    inside = signal["timestamp"].between(*WINDOW)
    return signal.loc[inside, ["timestamp", "baseline", "noise", "event_cost", "signal"]]


def draw(table: pd.DataFrame) -> plt.Figure:
    """Draw the background above and the meter beside it below."""
    time = pd.to_datetime(table["timestamp"])
    realised = table["baseline"] + table["noise"]
    figure, (upper, lower) = plt.subplots(2, 1, figsize=(5.58, 4.85), sharex=True)

    upper.plot(time, realised, color=NOISE, lw=1.2,
               label="background as generated")
    upper.plot(time, table["baseline"], color=BACK, lw=1.6, linestyle="--",
               label="predictable part, shape times 85")
    upper.set_ylim(0, 225)
    upper.set_ylabel("energy per half hour")
    upper.legend(frameon=False, fontsize=10, loc="lower left", ncol=2,
                 bbox_to_anchor=(0, 1.0, 1, 0.12), mode="expand", borderaxespad=0.2,
                 handlelength=1.4, handletextpad=0.4, columnspacing=0.8)
    upper.grid(alpha=0.25)

    lower.plot(time, table["signal"], color=METER, lw=1.2, label="meter total")
    lower.plot(time, table["baseline"], color=BACK, lw=1.6, linestyle="--",
               label="predictable background")
    lower.set_yscale("log")
    lower.set_ylabel("energy per half hour (log)")
    lower.legend(frameon=False, fontsize=10, loc="lower left", ncol=2,
                 bbox_to_anchor=(0, 1.0, 1, 0.12), mode="expand", borderaxespad=0.2,
                 handlelength=1.4, handletextpad=0.4, columnspacing=0.8)
    lower.grid(alpha=0.25, which="both")

    lower.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    lower.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    lower.tick_params(axis="x", labelsize=10)
    figure.tight_layout()
    return figure


def main() -> None:
    """Build the window, save it, and draw the figure from the saved table."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    window_signal().to_csv(TABLE, index=False)
    figure = draw(pd.read_csv(TABLE))
    for folder in ["images", "results_event_state"]:
        path = ROOT / folder / "signal_example.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path.relative_to(ROOT)}")
    print(f"saved -> {TABLE.relative_to(ROOT)}")
    plt.close(figure)


if __name__ == "__main__":
    main()
