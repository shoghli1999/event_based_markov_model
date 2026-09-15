"""
merged_scope_check.py
---------------------
What happens if the five inseparable activities are merged instead of removed.

The question
------------
The whole system keeps 35 activities. Seven are left out: five SRM activities
that always occur together in two groups, so their individual costs cannot be
separated, and two activities that never occur before the chronological cut.

Removing the five costs 4,896 events. The alternative is to keep them as two
group activities, one per group. Each group is then identifiable, the training
matrix has full rank 37, and only the two late activities are dropped. This
script measures what that alternative gains and what it costs.

What is compared
----------------
35 learnable    the scope used in the thesis
37 merged       the same log, with the five co-occurring activities kept as
                two group activities

Both scopes leave out the two late activities, because with no training events
their costs can only come out as zero.

For each scope the script reports
  * the activity cost error over the 30 noise seeds used elsewhere, over all of
    the scope's activities and over the 35 activities both scopes share, since
    only the second figure compares like with like;
  * the recovered cost of each merged group against its true value;
  * the decoding results of a single run, evaluated exactly as
    event_state_hmm.py evaluates every other scope.

The merged scope holds more events, so its chronological cut falls three
intervals earlier. Every event keeps the energy it was generated with; only its
activity name changes, so the true cost of a group is the mean energy of the
events it holds. A group is therefore not an activity, which is why the thesis
keeps the 35 individually identifiable ones.

Output
------
    results_event_state/merged_scope_check.csv

Usage
-----
    python final_model/merged_scope_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "final_model"))

import data_pipeline as pipeline  # noqa: E402
import event_state_hmm as model  # noqa: E402
from scope_progression import COLLINEAR_GROUPS  # noqa: E402

LATE_ACTIVITIES = {
    activity for activity, reason in pipeline.EXCLUDED_ACTIVITIES.items()
    if reason.startswith("absent")
}
GROUPS = sorted(set(COLLINEAR_GROUPS.values()))
MERGED = "merged"


def merged_events() -> pd.DataFrame:
    """The simulated log without the late activities, the five others renamed to their group."""
    events = model._simulated_complete_events()
    events = events[~events["activity"].isin(LATE_ACTIVITIES)].copy()
    events["activity"] = events["activity"].map(lambda a: COLLINEAR_GROUPS.get(a, a))
    return events.reset_index(drop=True)


def load_merged_problem() -> model.Problem:
    """Build the merged scope with the same steps load_problem uses for every other scope."""
    events = merged_events()
    activities = sorted(events["activity"].unique())
    signal, matrix = pipeline.build_signal_and_matrix(events)
    timeline = pd.DatetimeIndex(pd.to_datetime(matrix["timestamp"]))

    events = events.sort_values(["case", "ts"]).reset_index(drop=True)
    events["bin"] = timeline.get_indexer(events["ts"].dt.floor(pipeline.FREQ))
    gap_minutes = events.groupby("case", sort=False)["ts"].diff().dt.total_seconds()
    events["gap"] = np.log1p(gap_minutes.fillna(0.0).clip(lower=0.0) / 60.0)
    state_index = {activity: index for index, activity in enumerate(activities)}
    events["state"] = events["activity"].map(state_index).astype(int)

    event_cost = signal["event_cost"].to_numpy(float)
    X = matrix[activities].to_numpy(float)
    truth = (
        events.groupby("activity")["energy"].mean().reindex(activities).to_numpy(float)
    )
    return model.Problem(
        MERGED,
        events,
        activities,
        X,
        event_cost,
        event_cost + signal["noise"].to_numpy(float),
        truth,
        model.event_count_split(X),
        len(model._simulated_complete_events()),
        pipeline._duration_share(events, events["ts"].dt.floor(pipeline.FREQ)),
        signal["shape"].to_numpy(float),
    )


def seed_costs(problem: model.Problem, shared: list[str]) -> pd.DataFrame:
    """Cost error per noise seed, over all activities and over the shared 35."""
    columns = [problem.activities.index(activity) for activity in shared]
    rows = []
    for seed in model.ROBUSTNESS_SEEDS:
        noisy = model.with_noise(problem, seed)
        ols = model._fit_rewards(noisy)
        weighted, *_ = model._fit_aggregate_emissions(noisy)
        ols_error = np.abs(ols - noisy.truth)
        weighted_error = np.abs(weighted - noisy.truth)
        row = {
            "noise_seed": seed,
            "ols": float(ols_error.mean()),
            "weighted": float(weighted_error.mean()),
            "ols_shared": float(ols_error[columns].mean()),
            "weighted_shared": float(weighted_error[columns].mean()),
        }
        for group in GROUPS:
            if group in problem.activities:
                row[group] = float(weighted[problem.activities.index(group)])
        rows.append(row)
    return pd.DataFrame(rows).set_index("noise_seed")


def main() -> None:
    """Compare the thesis scope with the merged scope and save one row per scope."""
    learnable = model.load_problem("learnable")
    merged = load_merged_problem()
    problems = {"learnable": learnable, MERGED: merged}
    shared = list(learnable.activities)
    costs = {scope: seed_costs(problem, shared) for scope, problem in problems.items()}

    # evaluate_scope loads its own problem by name, so it is pointed at the two
    # problems built above for the length of this comparison and then restored.
    original = model.load_problem
    model.load_problem = lambda scope, spread_duration=True: problems.get(scope) or original(
        scope, spread_duration
    )
    try:
        decoded = {
            scope: model.evaluate_scope(scope, experts=0)[0].iloc[0] for scope in problems
        }
    finally:
        model.load_problem = original

    rows = []
    for scope, problem in problems.items():
        cost, run = costs[scope], decoded[scope]
        row = {
            "scope": "35 learnable (thesis)" if scope == "learnable" else "37 co-occurring merged",
            "activities": len(problem.activities),
            "events": len(problem.events),
            "training_rank": model.training_rank(problem),
            "ols_cost_error": float(cost["ols"].median()),
            "weighted_cost_error": float(cost["weighted"].median()),
            "weighted_wins_of_30": int((cost["weighted"] < cost["ols"]).sum()),
            "ols_cost_error_shared35": float(cost["ols_shared"].median()),
            "weighted_cost_error_shared35": float(cost["weighted_shared"].median()),
            "shared35_worse_than_thesis_seeds": int(
                (cost["weighted_shared"] > costs["learnable"]["weighted_shared"]).sum()
            ),
            "reconstruction_rmse": float(run["weighted_regression_test_rmse"]),
            "represented_test_events_percent": float(run["represented_test_events_percent"]),
            "variant_state_accuracy": float(run["variant_decoder_state_accuracy"]),
            "variant_event_energy_mae": float(run["variant_decoder_event_energy_mae"]),
            "pooled_state_accuracy": float(run["pooled_decoder_state_accuracy"]),
            "no_transition_state_accuracy": float(run["independent_decoder_state_accuracy"]),
        }
        for group in GROUPS:
            if group in problem.activities:
                key = group.lower().replace(" ", "_")
                row[f"{key}_true_cost"] = float(problem.truth[problem.activities.index(group)])
                row[f"{key}_weighted_median"] = float(cost[group].median())
        rows.append(row)

    table = pd.DataFrame(rows)
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = model.OUT_DIR / "merged_scope_check.csv"
    table.to_csv(path, index=False)
    print(table.T.to_string(header=False, float_format=lambda v: f"{v:,.6f}"))
    print(f"\nsaved -> {path.relative_to(model.OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
