"""
reward_layer.py
───────────────
Markov reward process on top of the event-state HMM.
Follows the design of markov_reward.py from the repository history.

What it adds
────────────
The event-state HMM learns how much each activity costs. This file combines
those costs with the process structure to answer a question regression cannot
answer at all:

    what is the expected energy of one COMPLETE case?

Why a case needs an END
───────────────────────
A transition table whose rows each sum to 1 says "after every activity another
activity always follows", so the case never finishes and expected visits are
infinite. Real cases do finish, so the chain needs an absorbing END state.

Two kinds of chain, and why both
────────────────────────────────
per variant   A variant is one fixed activity sequence, so its END is definite:
              variant 1 always ends on Clear Invoice, variant 3 always ends on
              Record Goods Receipt. For a fixed sequence the expected visits are
              simply how many times each activity appears in it. Exact, no
              matrix needed.

pooled        All variants mixed into one chain. Here paths branch and the END
              probability is shared out, so expected visits need the fundamental
              matrix N = (I - Q)^-1 of the absorbing chain.

The check
─────────
The pooled chain must reproduce the case-weighted average of the per-variant
chains. Those are two different calculations, so agreement is real evidence that
the pooled transition matrix is consistent with the individual process paths.
This is the same validation the original script used.

Everything is counted from cases that finish before the training cut, so no
test-period ordering enters the chain. One hidden state is still one activity;
this file only reads what the model already learned.

Training agreement is not a test
────────────────────────────────
A Markov chain built by counting always reproduces the average visits of the
cases it was counted from; that agreement is an identity, so it checks the code
and nothing else. The honest test is on cases the chain never saw, so this file
also reports every number on FUTURE cases, meaning cases that begin only after
the cut.

Future cases need one correction. The log is a recording with an end date, and a
case that starts near that end has little time left. A long case starting then is
still unfinished when the recording stops, so it never appears as a complete
path; only the quick cases survive to be seen. Late-starting cases are therefore
biased towards short ones, and their measured energy is too low for a reason that
has nothing to do with the model. Measured on top3: cases with room to spare
average 4.5984 events, late starters only 3.6314, and case length correlates
+0.43 with how much room was left.

The "settled" columns therefore keep only future cases that began at least the
90th percentile of training case duration before the recording ends, and the
"longest_follow_up" columns keep the quarter of future cases watched longest.
Neither group is a set of cases known to have finished; the first had a full
margin of observation and the second had the most of it. Where the strict group
exists and agrees with the full future group, the end of the recording explains
nothing. Where it cannot be formed at all, as on the whole system, the strict
test is simply unavailable, and the gap between the full future group and the
longest-watched quarter is reported as evidence CONSISTENT WITH truncation by
the end of the recording. It is not proof of truncation, and it is never process
drift on the strength of these numbers alone.

Usage
─────
    python final_model/reward_layer.py
    python final_model/reward_layer.py --scope top3 learnable
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_state_hmm import (  # noqa: E402
    FREQ,
    _complete_training_cases,
    _fit_rewards,
    _simulated_complete_events,
    fit_pooled_hmm,
    load_problem,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results_event_state"
SMOOTHING = 1e-9


def training_cases(problem) -> pd.DataFrame:
    """Only cases that finish before the cut, so the chain sees no test ordering."""
    complete = _complete_training_cases(problem.events, problem.cut)
    return problem.events[problem.events["case"].isin(complete)]


def variant_visits(problem) -> pd.DataFrame:
    """Expected visits per activity for every distinct training variant.

    A variant is a fixed sequence, so a case following it visits each activity a
    known number of times. Counting them is exact; no chain arithmetic is needed
    and the END is unambiguous because the sequence has a last element.
    """
    K = len(problem.activities)
    sequences = training_cases(problem).groupby("case", sort=False)["state"].apply(tuple)
    counts = sequences.value_counts()

    rows = []
    for sequence, case_count in counts.items():
        visits = np.bincount(np.asarray(sequence, dtype=int), minlength=K).astype(float)
        rows.append({
            "cases": int(case_count),
            "length": len(sequence),
            "ends_on": problem.activities[sequence[-1]],
            "visits": visits,
        })
    return pd.DataFrame(rows)


def pooled_visits(problem) -> np.ndarray:
    """Expected visits per activity from one pooled absorbing chain.

    Transitions are counted across all training cases, and each case's final
    activity contributes one step into END. That makes the activity-to-activity
    block Q substochastic, so (I - Q) can be inverted.
    """
    K = len(problem.activities)
    start = np.full(K, SMOOTHING)
    trans = np.full((K, K + 1), SMOOTHING)          # last column is END

    for _, case in training_cases(problem).groupby("case", sort=False):
        states = case["state"].to_numpy(int)
        if not len(states):
            continue
        start[states[0]] += 1.0
        np.add.at(trans, (states[:-1], states[1:]), 1.0)
        trans[states[-1], K] += 1.0                  # the case ends here

    start /= start.sum()
    trans /= trans.sum(axis=1, keepdims=True)
    Q = trans[:, :K]
    return start @ np.linalg.inv(np.eye(K) - Q)


SETTLED_QUANTILE = 0.90
# The last recorded event is not a safe end date for the recording. The log runs
# densely to 18 January 2019 and then stops. Exactly seven events out of
# 1,595,603 sit behind gaps of two weeks to five months after that, and taking
# the maximum stretches the observation window by ten months, which lets every
# future case appear to have had time to finish. The end is therefore the last
# event before the first gap longer than a week. There are only four such gaps in
# the whole log and all four lie in that stray tail, so the rule is unambiguous.
RECORDING_GAP = pd.Timedelta(days=7)
# Share of future cases kept for the longest follow-up check.
FOLLOW_UP_SHARE = 0.25


@lru_cache(maxsize=1)
def recording_end_timestamp() -> pd.Timestamp:
    """When the log stops recording, taken once from the complete log.

    One date is used for every scope. The recording period is a property of the
    log, not of the subset being analysed, so deriving it per scope would give
    each one a different observation window and make the settled columns
    incomparable.
    """
    stamps = pd.DatetimeIndex(_simulated_complete_events()["ts"]).sort_values()
    gaps = stamps[1:] - stamps[:-1]
    big = np.flatnonzero(gaps > RECORDING_GAP)
    return stamps[big[0]] if len(big) else stamps[-1]


def _observation_deadline(problem) -> pd.Timestamp:
    """The latest a case may start and still have a full margin of observation.

    Everything here is done in clock time rather than in interval indices. A
    scope holding fewer activities has a shorter timeline of its own, so an index
    would be capped at that scope's last interval and each scope would end up
    judged against a slightly different window. Timestamps keep the promise that
    the observation period is one and the same for every scope.
    """
    grouped = problem.events.groupby("case", sort=False)["ts"]
    first, last = grouped.min(), grouped.max()
    training = list(_complete_training_cases(problem.events, problem.cut))
    margin = (last[training] - first[training]).quantile(SETTLED_QUANTILE)
    return recording_end_timestamp() - margin


def future_cases(problem) -> set:
    """Cases that begin only after the cut, so the chain has never seen them."""
    first = problem.events.groupby("case", sort=False)["bin"].min()
    return set(first[first >= problem.cut].index)


def settled_future_cases(problem) -> set:
    """Future cases that began early enough to have finished before the log ends.

    The log stops on a fixed date. A case starting close to that date is still
    running when the recording stops, so its final steps were never written down
    and it looks shorter than it is. The margin is the 90th percentile of how
    long a training case takes, so nine out of ten cases had room to finish.
    """
    first = problem.events.groupby("case", sort=False)["ts"].min()
    deadline = _observation_deadline(problem)
    return {case for case in future_cases(problem) if first[case] <= deadline}


def longest_follow_up_cases(problem) -> set:
    """The quarter of future cases with the most observation time after they start.

    On the whole system no future case clears the strict margin above, so that
    column is empty there. This softer group is not a set of cases known to have
    finished; it is the group that was watched longest, and it is reported as
    supporting evidence rather than as a guaranteed result.
    """
    first = problem.events.groupby("case", sort=False)["ts"].min()
    future = sorted(future_cases(problem))
    if not future:
        return set()
    room = (recording_end_timestamp() - first[future]).clip(lower=pd.Timedelta(0))
    threshold = room.quantile(1.0 - FOLLOW_UP_SHARE)
    return set(room[room >= threshold].index)


def _case_energy(events: pd.DataFrame, cases: set) -> tuple[float, float, int]:
    """Measured mean energy, mean length and count for one group of cases."""
    part = events[events["case"].isin(cases)]
    if part.empty:
        return float("nan"), float("nan"), 0
    per_case = part.groupby("case")["energy"].sum()
    return float(per_case.mean()), float(len(part) / per_case.size), int(per_case.size)


def report(scope: str) -> dict:
    """Per-case energy from the variant chains, the pooled chain, and reality."""
    problem = load_problem(scope)
    model = fit_pooled_hmm(problem)
    ols_energy = _fit_rewards(problem)

    variants = variant_visits(problem)
    total_cases = int(variants["cases"].sum())
    stacked = np.vstack(variants["visits"].to_numpy())
    weights = variants["cases"].to_numpy(float)

    # case-weighted average over the per-variant chains
    weighted_visits = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
    pooled = pooled_visits(problem)

    measured = float(
        training_cases(problem).groupby("case")["energy"].sum().mean()
    )

    row = {
        "scope": scope,
        "activities": len(problem.activities),
        "training_cases": total_cases,
        "distinct_variants": len(variants),
        "measured_mean_case_energy": measured,
        "events_per_case_variants": float(weighted_visits.sum()),
        "events_per_case_pooled": float(pooled.sum()),
        "case_energy_variants_hmm": float(weighted_visits @ model.energy),
        "case_energy_pooled_hmm": float(pooled @ model.energy),
        "case_energy_variants_true": float(weighted_visits @ problem.truth),
        "case_energy_variants_ols": float(weighted_visits @ ols_energy),
    }
    row["pooled_vs_variants_gap"] = abs(
        row["case_energy_pooled_hmm"] - row["case_energy_variants_hmm"]
    )
    for name in ("true", "hmm", "ols"):
        key = f"case_energy_variants_{name}"
        row[f"{name}_error_percent"] = 100.0 * abs(row[key] - measured) / measured

    # The same prediction, now on cases the chain never saw.
    predicted = row["case_energy_variants_hmm"]
    for label, group in (("future", future_cases(problem)),
                         ("settled", settled_future_cases(problem)),
                         ("longest_follow_up", longest_follow_up_cases(problem))):
        energy, length, count = _case_energy(problem.events, group)
        row[f"{label}_cases"] = count
        row[f"{label}_measured_mean_case_energy"] = energy
        row[f"{label}_events_per_case"] = length
        row[f"{label}_hmm_error_percent"] = (
            100.0 * abs(predicted - energy) / energy if count else float("nan")
        )
    return row


def main(scopes: list[str]) -> None:
    """Report expected case energy for each scope and save the table."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope in scopes:
        print(f"\n--- {scope} ---")
        row = report(scope)
        rows.append(row)
        print(f"  distinct training variants : {row['distinct_variants']:,}")
        print(f"  events per case            : {row['events_per_case_variants']:.4f} "
              f"(variants)  {row['events_per_case_pooled']:.4f} (pooled)")
        print(f"  measured case energy       : {row['measured_mean_case_energy']:.4f}")
        print(f"  from variant chains + HMM  : {row['case_energy_variants_hmm']:.4f}"
              f"   ({row['hmm_error_percent']:.2f}% off)")
        print(f"  from pooled chain + HMM    : {row['case_energy_pooled_hmm']:.4f}")
        print(f"  pooled vs variants gap     : {row['pooled_vs_variants_gap']:.4f}")
        print(f"  --- the same prediction on cases the chain never saw ---")
        print(f"  future cases               : {row['future_cases']:,}"
              f"   measured {row['future_measured_mean_case_energy']:.4f}"
              f"   ({row['future_hmm_error_percent']:.2f}% off)")
        print(f"  of those, settled ones     : {row['settled_cases']:,}"
              f"   measured {row['settled_measured_mean_case_energy']:.4f}"
              f"   ({row['settled_hmm_error_percent']:.2f}% off)")
        print(f"  events per case            : {row['events_per_case_variants']:.4f} training"
              f"   {row['future_events_per_case']:.4f} future"
              f"   {row['settled_events_per_case']:.4f} settled")

    table = pd.DataFrame(rows)
    path = OUT_DIR / "reward_layer.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved -> {path.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markov reward layer on the event-state HMM.")
    parser.add_argument("--scope", nargs="+", default=["top3", "learnable"])
    main(parser.parse_args().scope)
