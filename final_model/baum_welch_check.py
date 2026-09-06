"""
baum_welch_check.py
───────────────────
Evidence for switching Baum-Welch off in the emission fit.

The question
────────────
The model learns one energy cost per activity from 30-minute meter totals.
Two estimators are available for that:

  weighted regression  reads the measured interval totals directly;
  Baum-Welch           refines the same costs event by event.

Baum-Welch is the standard choice for a hidden Markov model, so it has to be
tested rather than assumed.

Why it cannot help here
───────────────────────
Baum-Welch exists to recover states that nobody observed. In this model a
state is an activity, and the event log names the activity of every training
event. So at training time nothing is hidden. Baum-Welch throws those names
away and re-infers them from values, which can only lose information that was
already correct.

It is also fed invented data. A hidden Markov model needs one number per
event, but the meter only reports one total per interval. The code therefore
gives every event its current estimated cost plus an equal share of that
interval's leftover. The leftover is mostly meter noise, so the invented
per-event numbers carry no extra information.

What this script measures
─────────────────────────
For each scope it fits the model twice, once with Baum-Welch off and once with
it on, and reports the mean absolute activity-cost error against the known
truth. Nothing else changes.

Baum-Welch stays in event_state_hmm.py and can be switched back on with
BAUM_WELCH_DEFAULT or the baum_welch argument of fit_pooled_hmm.

Output
──────
    results_event_state/baum_welch_check.csv

Usage
─────
    python final_model/baum_welch_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import event_state_hmm as model  # noqa: E402

SCOPES = ["top1", "top2", "top3", "top5", "learnable"]


def cost_error(problem, baum_welch: bool) -> tuple[float, bool]:
    """Mean absolute activity-cost error for one setting of Baum-Welch."""
    fitted = model.fit_pooled_hmm(problem, baum_welch=baum_welch)
    return float(np.abs(fitted.energy - problem.truth).mean()), fitted.baum_welch_used


def main() -> None:
    """Fit every scope twice and save the Baum-Welch comparison."""
    rows = []
    for scope in SCOPES:
        problem = model.load_problem(scope)
        without, _ = cost_error(problem, baum_welch=False)
        with_bw, used = cost_error(problem, baum_welch=True)
        rows.append({
            "scope": scope,
            "activities": len(problem.activities),
            "cost_error_without_baum_welch": without,
            "cost_error_with_baum_welch": with_bw,
            "baum_welch_update_accepted": used,
            "change_percent": 100.0 * (with_bw - without) / without,
            "baum_welch_helps": with_bw < without,
        })
        print(pd.DataFrame(rows[-1:]).to_string(index=False))

    table = pd.DataFrame(rows)
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(model.OUT_DIR / "baum_welch_check.csv", index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nsaved {model.OUT_DIR / 'baum_welch_check.csv'}")
    print(
        f"Baum-Welch helps on {int(table['baum_welch_helps'].sum())} "
        f"of {len(table)} scopes."
    )


if __name__ == "__main__":
    main()
