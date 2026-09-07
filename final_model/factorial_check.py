"""
factorial_check.py
──────────────────
RQ4 asks for additive or factorial composition. This measures the factorial one.

Additive composition is what the decoder already does: many case-level chains
run at once and the meter sees their sum.

Factorial composition goes further. Each variant gets its own chain with its own
activity costs, so an activity appearing in two variants is allowed two different
costs. The meter still sees the sum. This script builds that model and compares
it with the shared model, where one cost per activity is used everywhere.

Output
──────
    results_event_state/factorial_check.csv

Usage
─────
    python final_model/factorial_check.py
    python final_model/factorial_check.py --scope top2 top3
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


def variant_blocks(problem) -> tuple[np.ndarray, int]:
    """One block of activity columns per variant, built with the same splitting."""
    paths = problem.events.groupby("case", sort=False)["state"].apply(tuple)
    order = list(paths.value_counts().index)
    rank = {path: index for index, path in enumerate(order)}
    variant = problem.events["case"].map(
        {case: rank[path] for case, path in paths.items()}
    ).to_numpy(int)

    interval = problem.events["ts"].dt.floor(pipeline.FREQ)
    timeline = pd.date_range(interval.min(), interval.max(), freq=pipeline.FREQ)
    where, which, weight = pipeline._split_events(problem.events, interval, timeline)

    activities = len(problem.activities)
    blocks = np.zeros((len(problem.target), len(order) * activities))
    state = problem.events["state"].to_numpy(int)
    column = variant[which] * activities + state[which]
    np.add.at(blocks, (where, column), weight)
    return blocks, len(order)


def compare(scope: str) -> dict:
    """Shared costs against one set of costs per variant."""
    problem = model.load_problem(scope)
    blocks, variants = variant_blocks(problem)
    activities = len(problem.activities)
    cut = problem.cut

    # Both sides use the same three-term variance model as the rest of the
    # thesis.  Leaving the background shape out here would weight the intervals
    # by a different rule from every other table, and the shared column would
    # then disagree with the same estimator reported elsewhere.
    background = problem.background_shape[:cut]
    shared, _, _ = model.weighted_fit(problem.X[:cut], problem.target[:cut], background)
    factorial, _, _ = model.weighted_fit(blocks[:cut], problem.target[:cut], background)
    truth = np.tile(problem.truth, variants)

    unsupported = int((blocks[:cut].sum(axis=0) == 0).sum())
    return {
        "scope": scope,
        "variants": variants,
        "activities": activities,
        "shared_costs": activities,
        "factorial_costs": variants * activities,
        "factorial_rank": int(np.linalg.matrix_rank(blocks[:cut])),
        "costs_with_no_training_data": unsupported,
        "shared_cost_error": float(np.abs(shared - problem.truth).mean()),
        "factorial_cost_error": float(np.abs(factorial - truth).mean()),
        "shared_test_rmse": model._rmse(
            problem.target, problem.X @ shared, cut
        ),
        "factorial_test_rmse": model._rmse(
            problem.target, blocks @ factorial, cut
        ),
    }


def main(scopes: list[str]) -> None:
    """Compare shared activity costs with one set of costs per variant."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [compare(scope) for scope in scopes]
    table = pd.DataFrame(rows)
    path = model.OUT_DIR / "factorial_check.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    print(f"\nsaved -> {path.relative_to(model.OUT_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factorial composition for RQ4.")
    parser.add_argument("--scope", nargs="+", default=["top2", "top3", "top5"])
    main(parser.parse_args().scope)
