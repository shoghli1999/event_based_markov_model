"""
interval_convention_check.py
────────────────────────────
An event lasts about three minutes and the meter reports every thirty, so an
event can start just before one interval ends and finish in the next. There are
three ways to place it, and this script measures all three on the same data.

    whole event     energy and activity count both charged to the interval the
                    event starts in.
    both shared     energy and activity count both split between the two
                    intervals, in proportion to time spent in each. This thesis.
    energy only     energy split, activity count left whole in the starting
                    interval. My supervisor's evaluation code describes the
                    matrix as counts of events per interval, so this is what
                    pairing his generator with that description would give.

"energy only" leaves energy in intervals whose count row is empty, and no
estimator can explain energy where nothing is recorded as happening. The last
column counts those intervals.

Output
──────
    results_event_state/interval_convention_check.csv

Usage
─────
    python final_model/interval_convention_check.py
    python final_model/interval_convention_check.py --scope top3 learnable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_pipeline as pipeline  # noqa: E402
import event_state_hmm as model  # noqa: E402

PLACEMENTS = ["whole event", "both shared (this thesis)", "energy only"]


def build(events: pd.DataFrame, placement: str):
    """Counts, meter signal, background shape, truth and event energy alone.

    ``cost`` is returned separately from ``target`` because the meter signal also
    carries background noise, which is non-zero in nearly every interval.  Asking
    whether an interval holds energy that no counted activity can explain has to
    be asked of the event energy alone, otherwise every event-free interval in
    the log answers yes.
    """
    activities = sorted(events["activity"].unique())
    interval = events["ts"].dt.floor(pipeline.FREQ)
    timeline = pd.date_range(interval.min(), interval.max(), freq=pipeline.FREQ)
    start = timeline.get_indexer(pd.DatetimeIndex(interval))
    code = events["activity"].map(
        {activity: index for index, activity in enumerate(activities)}
    ).to_numpy(int)
    energy = events["energy"].to_numpy(float)

    cost = np.zeros(len(timeline))
    counts = np.zeros((len(timeline), len(activities)))
    if placement == "whole event":
        np.add.at(cost, start, energy)
        np.add.at(counts, (start, code), 1.0)
    else:
        where, which, weight = pipeline._split_events(events, interval, timeline)
        np.add.at(cost, where, energy[which] * weight)
        if placement == "energy only":
            np.add.at(counts, (start, code), 1.0)
        else:
            np.add.at(counts, (where, code[which]), weight)

    shape = pipeline.sine_baseline(timeline)
    target = cost + pipeline.background_noise(shape, pipeline.SIGNAL_SEED)
    truth = events.groupby("activity")["energy"].mean().reindex(activities).to_numpy()
    return counts, target, shape, truth, cost


def compare(events: pd.DataFrame, scope: str, placement: str) -> dict:
    """Fit OLS and the weighted model on one placement and score both."""
    X, y, shape, truth, cost = build(events, placement)
    cut = model.event_count_split(X)
    ols = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)[0]
    weighted, _, _ = model.weighted_fit(X[:cut], y[:cut], shape[:cut])
    ols_error = float(np.abs(ols - truth).mean())
    weighted_error = float(np.abs(weighted - truth).mean())
    stranded = (cost > 1e-9) & (X.sum(axis=1) <= 1e-9)
    return {
        "scope": scope,
        "placement": placement,
        "ols_cost_error": ols_error,
        "weighted_cost_error": weighted_error,
        "weighted_beats_ols_percent": 100.0 * (ols_error - weighted_error) / ols_error,
        "intervals_with_energy_but_no_count": int(stranded.sum()),
        "stranded_energy_percent": 100.0 * float(cost[stranded].sum())
        / max(float(cost.sum()), 1e-9),
    }


def main(scopes: list[str]) -> None:
    """Fit every placement on every scope and save what the choice costs."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    simulated = model._simulated_complete_events()
    rows = []
    for scope in scopes:
        events = (
            pipeline.filter_learnable_events(simulated)[0] if scope == "learnable"
            else pipeline.select_coverage(simulated, scope.removeprefix("top"))
        )
        for placement in PLACEMENTS:
            rows.append(compare(events, scope, placement))

    table = pd.DataFrame(rows)
    path = model.OUT_DIR / "interval_convention_check.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    print(f"\nsaved -> {path.relative_to(model.OUT_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the three event placements.")
    parser.add_argument("--scope", nargs="+", default=["top3", "learnable"])
    main(parser.parse_args().scope)
