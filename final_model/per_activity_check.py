"""
per_activity_check.py
─────────────────────
The per-activity view that a total cannot hide.

Why this exists
───────────────
reward_layer.py predicts the energy of one complete case. That is a single
total, and a total can be right while the parts inside it are wrong: swap the
cost of two equally frequent activities and the total does not move at all.
This is the exact weakness the thesis is about, so it must not be the only
check.

This script opens the total up. For every activity it reports:

  visits_counted    how often a case really visits that activity, counted
                    directly from the training cases;
  visits_chain      how often the pooled absorbing Markov chain expects a case
                    to visit it. A gap here is a failure of the chain itself;
  energy_real       the energy that activity really contributes to one case;
  energy_predicted  visits_chain multiplied by the learned cost. A gap here is
                    a misattribution that the case total would hide;
  cost_true         the ground-truth cost of one event of that activity;
  cost_estimated    the learned cost, and how many training events supported it.

Output
──────
    results_event_state/per_activity_check.csv

Usage
─────
    python final_model/per_activity_check.py
    python final_model/per_activity_check.py --scope top3 learnable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "final_model"))

import event_state_hmm as model  # noqa: E402
from reward_layer import pooled_visits, training_cases, variant_visits  # noqa: E402


def per_activity(scope: str) -> pd.DataFrame:
    """One row per activity: visits, energy and cost, predicted against real."""
    problem = model.load_problem(scope)
    activity_count = len(problem.activities)

    variants = variant_visits(problem)
    stacked = np.vstack(variants["visits"].to_numpy())
    weights = variants["cases"].to_numpy(float)
    counted = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
    chain = pooled_visits(problem)

    train = training_cases(problem)
    cases = train["case"].nunique()
    real = (
        train.groupby("state")["energy"].sum()
        .reindex(range(activity_count)).fillna(0.0).to_numpy() / cases
    )
    estimated, *_ = model._fit_aggregate_emissions(problem)

    before_cut = problem.events[problem.events["bin"] < problem.cut]
    support = (
        before_cut["state"].value_counts()
        .reindex(range(activity_count)).fillna(0).astype(int).to_numpy()
    )

    table = pd.DataFrame({
        "scope": scope,
        "activity": problem.activities,
        "training_events": support,
        "visits_counted": counted,
        "visits_chain": chain,
        "energy_real": real,
        "energy_predicted": chain * estimated,
        "cost_true": problem.truth,
        "cost_estimated": estimated,
    })
    table["visit_gap"] = table["visits_chain"] - table["visits_counted"]
    table["energy_gap"] = table["energy_predicted"] - table["energy_real"]
    table["cost_error"] = np.abs(table["cost_estimated"] - table["cost_true"])
    return table.sort_values("energy_real", ascending=False).reset_index(drop=True)


def main(scopes: list[str]) -> None:
    """Open the case total up activity by activity and save the table."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = []
    for scope in scopes:
        table = per_activity(scope)
        tables.append(table)
        print(f"\n--- {scope} ---")
        print(table.head(8)[[
            "activity", "training_events", "visits_counted", "visits_chain",
            "energy_real", "energy_predicted",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  largest visit gap        : {table['visit_gap'].abs().max():.6f}")
        print(f"  largest energy gap       : {table['energy_gap'].abs().max():.4f}")
        print(f"  sum of energy gaps       : {table['energy_gap'].abs().sum():.4f}")
        print(f"  largest single cost error: {table['cost_error'].max():.4f} "
              f"({table.loc[table['cost_error'].idxmax(), 'activity']}, "
              f"{int(table.loc[table['cost_error'].idxmax(), 'training_events'])} training events)")

    combined = pd.concat(tables, ignore_index=True)
    path = model.OUT_DIR / "per_activity_check.csv"
    combined.to_csv(path, index=False)
    print(f"\nsaved -> {path.relative_to(model.OUT_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-activity view of the reward layer.")
    parser.add_argument("--scope", nargs="+", default=["top3", "learnable"])
    main(parser.parse_args().scope)
