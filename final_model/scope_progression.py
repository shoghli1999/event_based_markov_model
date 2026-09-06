"""
scope_progression.py
────────────────────
Why the whole-system experiment reports 35 activities and not 42.

The three scopes
────────────────
42  every activity in the cleaned log. Five of them always occur together at
    the same timestamps, so their individual costs cannot be recovered. Two more
    activities never appear before the training cut, so their costs cannot be
    learned either.

    Rank is measured from whole-event counts, one event in the interval it starts
    in. The model shares an event across the two intervals its duration covers,
    which is right for the energy, but it would separate two always-together
    activities by a hair of random duration and make the rank claim more than the
    data supports. Rank is a statement about the process, so it is measured from
    the process.

    results_event_state/cost_precision.csv holds the supporting detail: those
    five activities are estimated about fifty times less precisely than the rest,
    and two of them receive the identical estimate, which is simply their
    average.

39  the five co-occurring activities folded into their two groups. Nothing is
    dropped; every event is kept. The matrix becomes full rank over the 39
    columns, but the two cold-start activities are still unlearnable.

35  the five collinear activities and the two cold-start activities removed.
    Every remaining cost is individually recoverable. This costs 0.339% of the
    events.

Reporting all three is the honest way to present the identifiability limit: it
shows what was removed, why, and what it cost.

Output
──────
    results_event_state/scope_progression.csv

Usage
─────
    python final_model/scope_progression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import (  # noqa: E402
    EXCLUDED_ACTIVITIES,
    build_signal_and_matrix,
    filter_learnable_events,
)
from event_state_hmm import (  # noqa: E402
    _simulated_complete_events,
    event_count_split,
    weighted_fit,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results_event_state"

COLLINEAR_GROUPS = {
    "SRM: Awaiting Approval": "SRM group A",
    "SRM: Complete": "SRM group A",
    "SRM: Document Completed": "SRM group A",
    "SRM: Held": "SRM group B",
    "SRM: Incomplete": "SRM group B",
}


# The five activities above, as a set, so the grouping is written only once.
CO_OCCURRING = set(COLLINEAR_GROUPS)


def matrix_of(events: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Interval-by-activity count matrix and its column names.

    Each event is counted whole in the interval it starts in.  The model itself
    shares an event across the two intervals its duration covers, which is right
    for the energy, but it would separate two always-together activities by a
    hair of random duration and make the rank over-report what the data can tell
    apart.  Rank is a statement about the process, so it is measured this way.
    """
    _, matrix = build_signal_and_matrix(events, spread_duration=False)
    activities = [c for c in matrix.columns if c != "timestamp"]
    return matrix[activities].to_numpy(float), activities


def audit(name: str, events: pd.DataFrame, total_events: int) -> dict:
    """Rank of the whole matrix and of the training part, plus what is unlearnable."""
    X, activities = matrix_of(events)
    cut = event_count_split(X)
    cold_start = [
        a for i, a in enumerate(activities)
        if X[:cut, i].sum() == 0 and X[cut:, i].sum() > 0
    ]
    full_rank = int(np.linalg.matrix_rank(X))
    return {
        "scope": name,
        "activities": len(activities),
        "events": len(events),
        "event_percent": 100.0 * len(events) / total_events,
        "full_rank": full_rank,
        "training_rank": int(np.linalg.matrix_rank(X[:cut])),
        "rank_deficit": len(activities) - full_rank,
        "cold_start_activities": len(cold_start),
        "every_cost_recoverable": full_rank == len(activities) and not cold_start,
    }


def cost_precision(events: pd.DataFrame) -> pd.DataFrame:
    """Standard error of every estimated activity cost on the 42-activity scope.

    Rank counts dimensions; this counts how well each cost is actually pinned
    down.  A cost whose standard error is large is not recoverable however the
    rank is reported.
    """
    signal, matrix = build_signal_and_matrix(events)
    activities = [c for c in matrix.columns if c != "timestamp"]
    X_all = matrix[activities].to_numpy(float)
    y_all = (signal["event_cost"] + signal["noise"]).to_numpy(float)
    cut = event_count_split(X_all)
    X, y = X_all[:cut], y_all[:cut]

    energy, _, variance = weighted_fit(X, y)
    weight = 1.0 / variance
    # A near-singular column can leave a tiny negative value on the diagonal
    # through floating point, which would make the square root not-a-number.
    standard_error = np.sqrt(
        np.maximum(np.diag(np.linalg.pinv((X * weight[:, None]).T @ X)), 0.0)
    )
    truth = (
        events.groupby("activity")["energy"].mean().reindex(activities).to_numpy()
    )
    return pd.DataFrame({
        "activity": activities,
        "true_cost": truth,
        "estimate": energy,
        "standard_error": standard_error,
        "absolute_error": np.abs(energy - truth),
        "training_events": X.sum(axis=0),
        "co_occurring_group": [a in CO_OCCURRING for a in activities],
    })


def main() -> None:
    """Report why the whole system keeps 35 of 42 activities."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    complete = _simulated_complete_events()
    total = len(complete)

    # 39: relabel the collinear activities into their groups, drop nothing
    merged = complete.copy()
    merged["activity"] = merged["activity"].map(
        lambda a: COLLINEAR_GROUPS.get(a, a)
    )

    learnable, _ = filter_learnable_events(complete)

    table = pd.DataFrame([
        audit("42 all activities", complete, total),
        audit("39 collinear merged", merged, total),
        audit("35 learnable", learnable, total),
    ])
    table.to_csv(OUT_DIR / "scope_progression.csv", index=False)

    print("\nScope progression: what can be recovered, and what it costs")
    print("=" * 92)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    # Kept as quiet supporting evidence, in case the exclusion is questioned.
    precision = cost_precision(complete)
    precision.to_csv(OUT_DIR / "cost_precision.csv", index=False)

    print("\nExcluded from the 35-activity scope:")
    for activity, reason in sorted(EXCLUDED_ACTIVITIES.items()):
        print(f"  {activity:<34} {reason}")
    print(f"\nsaved -> {(OUT_DIR / 'scope_progression.csv').relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
