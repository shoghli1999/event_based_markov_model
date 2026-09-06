"""Load the BPI 2019 log and create the controlled energy experiment.

Only the event timestamps and activity names come from the real log.  Activity
energy, event duration, background load and measurement noise are synthetic.
The settings copied from my supervisor's work are marked below; the remaining
settings are this thesis's controlled experimental choices.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

# Settings inherited from my supervisor's work.
FREQ = "30min"
DURATION_MEAN = 180.0
DURATION_STD = 20.0
DURATION_MIN = 120.0
DURATION_MAX = 240.0
DURATION_SCALE = 0.01
SINE_PEAK_HOUR = 14.0
SINE_AMPLITUDE = 1.0
SINE_SHIFT = 1.1
# My supervisor multiplies the background shape by a fresh random number drawn
# from this range in every interval, so the background is both large and
# genuinely noisy rather than a smooth curve.
BACKGROUND_RANGE = (80.0, 90.0)

# Controlled choices made for this thesis.
SIGNAL_SEED = 123
DURATION_SEED = 42

# The activity values come from my supervisor's work.  Four extra CSV spellings
# are explicit so that no cost is silently invented when names differ.
BASE_COSTS = {
    "Record Goods Receipt": 99,
    "Create Purchase Order Item": 98,
    "Record Invoice Receipt": 97,
    "Vendor creates invoice": 54,
    "Clear Invoice": 56,
    "Record Service Entry Sheet": 81,
    "Remove Payment Block": 59,
    "Create Purchase Requisition Item": 11,
    "Receive Order Confirmation": 66,
    "Change Quantity": 2,
    "Change Price": 1,
    "Delete Purchase Order item": 8,
    "Cancel Invoice Receipt": 9,
    "Change Approval for Purchase Order": 21,
    "Vendor creates debit memo": 20,
    "Change Delivery Indicator": 52,
    "Cancel Goods Receipt": 10,
    "Release Purchase Order": 60,
    "SRM: In Transfer to Execution Syst.": 87,
    "SRM: Created": 78,
    "SRM: Complete": 77,
    "SRM: Awaiting Approval": 63,
    "SRM Document Completed": 70,
    "SRM: Ordered": 44,
    "SRM Change was Transmitted": 92,
    "Reactivate Purchase Order Item": 57,
    "Block Purchase Order Item": 31,
    "Cancel Subsequent Invoice": 32,
    "Change Storage Location": 36,
    "Update Order Confirmation": 22,
    "Record Subsequent Invoice": 40,
    "Release Purchase Requisition": 62,
    "Set Payment Block": 26,
    "SRM: Deleted": 5,
    "Change Currency": 150,
    "Change Final Invoice Indicator": 144,
    "SRM: Transaction Completed": 121,
    "SRM: Incomplete": 111,
    "SRM: Held": 88,
    "Change payment term": 4,
    "Change Rejection Indicator": 200,
    "Delete Purchase Order Item": 8,
    "SRM: Change was Transmitted": 92,
    "SRM: Document Completed": 70,
    "SRM: Transfer Failed (E.Sys.)": 100,
}

EXCLUDED_ACTIVITIES = {
    "SRM: Awaiting Approval": "co-occurring SRM group, not separately identifiable",
    "SRM: Complete": "co-occurring SRM group, not separately identifiable",
    "SRM: Document Completed": "co-occurring SRM group, not separately identifiable",
    "SRM: Held": "co-occurring SRM group, not separately identifiable",
    "SRM: Incomplete": "co-occurring SRM group, not separately identifiable",
    "Release Purchase Requisition": "absent from chronological training data",
    "SRM: Transfer Failed (E.Sys.)": "absent from chronological training data",
}

CSV_COLUMNS = {
    "case concept:name": "case",
    "event concept:name": "activity",
    "event time:timestamp": "ts",
}


def _csv_path() -> Path:
    """Find the local CSV export without hiding which file is used."""
    candidates = []
    configured = os.environ.get("BPI2019_CSV")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        ROOT / "BPI_Challenge_2019.csv",
        Path(
            "/Users/shirin/Documents/shirin/passau/thesis/code/"
            "BPI_Challenge_2019/BPI_Challenge_2019.csv"
        ),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "BPI_Challenge_2019.csv was not found. Put it beside this script or "
        "set the BPI2019_CSV environment variable."
    )


def load_events() -> pd.DataFrame:
    """Read the three required columns and remove invalid placeholder years."""
    events = pd.read_csv(
        _csv_path(),
        usecols=list(CSV_COLUMNS),
        dtype=str,
        encoding="cp1252",
    ).rename(columns=CSV_COLUMNS)
    events["ts"] = pd.to_datetime(events["ts"], dayfirst=True, errors="coerce")
    valid = events["ts"].notna() & events["ts"].dt.year.between(2018, 2019)
    dropped = int((~valid).sum())
    events = events.loc[valid].sort_values(["case", "ts"]).reset_index(drop=True)
    print(f"  loaded {len(events):,} events; dropped {dropped} invalid timestamps")
    return events


def select_coverage(events: pd.DataFrame, number: str) -> pd.DataFrame:
    """Keep the cases belonging to the N most frequent complete paths."""
    paths = events.groupby("case", sort=False)["activity"].apply(tuple)
    selected = set(paths.value_counts().head(int(number)).index)
    cases = set(paths[paths.isin(selected)].index)
    result = events[events["case"].isin(cases)].reset_index(drop=True)
    print(
        f"  top-{number}: {result['case'].nunique():,} cases, "
        f"{len(result):,} events"
    )
    return result


def filter_learnable_events(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude only the co-occurring group and the chronological cold-start ones.

    The five SRM activities always appear together at the same timestamps, so
    their count columns carry the same information and no estimator can split
    their costs.  Sharing an event across intervals makes the matrix look full
    rank, but only because each event was given a random synthetic duration;
    final_model/scope_progression.py shows the rank returning to 39 when every
    duration is the same, so the exclusion stands under both conventions.
    """
    counts = events["activity"].value_counts()
    audit = pd.DataFrame([
        {
            "activity": activity,
            "reason": reason,
            "events_excluded": int(counts.get(activity, 0)),
        }
        for activity, reason in EXCLUDED_ACTIVITIES.items()
    ]).sort_values("activity")
    retained = events[~events["activity"].isin(EXCLUDED_ACTIVITIES)].copy()
    retained = retained.reset_index(drop=True)
    audit["complete_log_events"] = len(events)
    audit["retained_events"] = len(retained)
    audit["retained_percent"] = 100.0 * len(retained) / len(events)
    return retained, audit.reset_index(drop=True)


def base_cost_table(activities: list[str]) -> dict[str, float]:
    """Return explicit costs and fail if a name has no documented value."""
    missing = sorted(set(activities) - set(BASE_COSTS))
    if missing:
        raise KeyError(f"No explicit synthetic cost for: {missing}")
    return {activity: float(BASE_COSTS[activity]) for activity in activities}


def simulate(events: pd.DataFrame, costs: dict[str, float]) -> pd.DataFrame:
    """Attach reproducible synthetic duration and energy to every event."""
    rng = np.random.default_rng(DURATION_SEED)
    duration = np.clip(
        rng.normal(DURATION_MEAN, DURATION_STD, len(events)),
        DURATION_MIN,
        DURATION_MAX,
    )
    result = events.copy()
    result["duration"] = duration
    result["energy"] = (
        result["activity"].map(costs).to_numpy(float)
        + duration * DURATION_SCALE
    )
    return result


def sine_baseline(timeline: pd.DatetimeIndex) -> np.ndarray:
    """Daily background shape with its maximum at 14:00, before scaling."""
    hours = np.asarray(timeline.hour + timeline.minute / 60.0, dtype=float)
    return SINE_SHIFT + SINE_AMPLITUDE * np.sin(
        2.0 * np.pi * (hours - SINE_PEAK_HOUR + 6.0) / 24.0
    )


def background_noise(shape: np.ndarray, seed: int) -> np.ndarray:
    """The part of the background that cannot be predicted from the clock.

    My supervisor multiplies the shape by a fresh random number in every
    interval.  The average of that range is predictable from the time of day and
    is removed, leaving real noise that is larger at the hours when the shape is
    high and smaller at night.
    """
    low, high = BACKGROUND_RANGE
    draw = np.random.default_rng(seed).uniform(low, high, len(shape))
    return shape * (draw - 0.5 * (low + high))


def _duration_share(events: pd.DataFrame, interval: pd.Series) -> np.ndarray:
    """Fraction of each event's duration that falls in the following interval."""
    step = pd.Timedelta(FREQ).total_seconds()
    duration = events["duration"].to_numpy(float)
    if (duration > step).any():
        raise ValueError("an event longer than one interval would need more shares")
    into = (events["ts"] - interval).dt.total_seconds().to_numpy()
    return np.clip(into + duration - step, 0.0, duration) / duration


def _split_events(
    events: pd.DataFrame, interval: pd.Series, timeline: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut every event into the one or two intervals its duration covers.

    Returns, for each piece, which interval it lands in, which event it came
    from, and what fraction of that event it is.  Energy and activity counts are
    both built from this one split, so they can never disagree.  A piece that
    would fall past the last interval has nowhere to go and is dropped.
    """
    share_next = _duration_share(events, interval)
    start = timeline.get_indexer(pd.DatetimeIndex(interval))
    event = np.arange(len(start))
    following = start + 1
    inside = following < len(timeline)
    return (
        np.concatenate([start, following[inside]]),
        np.concatenate([event, event[inside]]),
        np.concatenate([1.0 - share_next, share_next[inside]]),
    )


def _spread_signal_and_matrix(
    events: pd.DataFrame,
    interval: pd.Series,
    timeline: pd.DatetimeIndex,
    activities: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Energy and activity counts shared across intervals, as my supervisor does."""
    where, which, weight = _split_events(events, interval, timeline)

    energy = events["energy"].to_numpy(float)
    event_cost = np.zeros(len(timeline))
    np.add.at(event_cost, where, energy[which] * weight)

    column = {activity: index for index, activity in enumerate(activities)}
    code = events["activity"].map(column).to_numpy(int)
    counts = np.zeros((len(timeline), len(activities)))
    np.add.at(counts, (where, code[which]), weight)

    matrix = pd.DataFrame(counts, index=timeline, columns=activities)
    matrix.index.name = "timestamp"
    return event_cost, matrix


def build_signal_and_matrix(
    events: pd.DataFrame,
    spread_duration: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate event counts and energy onto one 30-minute timeline.

    An event lasting about three minutes can cross a 30-minute boundary.  Its
    energy and its activity count are then shared between the two intervals in
    proportion to the time spent in each, following my supervisor's generator.
    Both sides move together, so the signal and the count matrix always describe
    the same thing.

    ``spread_duration=False`` charges the whole event to the interval it starts
    in.  The pipeline never uses it; it exists only so
    final_model/interval_convention_check.py can measure the difference.
    """
    interval = events["ts"].dt.floor(FREQ)
    timeline = pd.date_range(interval.min(), interval.max(), freq=FREQ)
    if spread_duration:
        activities = sorted(events["activity"].unique())
        event_cost, matrix = _spread_signal_and_matrix(
            events, interval, timeline, activities
        )
    else:
        event_cost = (
            events.assign(interval=interval)
            .groupby("interval")["energy"]
            .sum()
            .reindex(timeline, fill_value=0.0)
            .to_numpy(float)
        )
        matrix = (
            events.assign(interval=interval)
            .groupby(["interval", "activity"])
            .size()
            .unstack(fill_value=0)
            .reindex(timeline, fill_value=0)
            .astype(int)
        )
        matrix.index.name = "timestamp"
    # Guard against the two sides drifting apart.  If an interval holds energy
    # but no counted activity, no estimator can ever explain it, and every error
    # measured afterwards is meaningless.  This is exactly what happens if the
    # energy is shared across intervals while the counts are not.
    counted = matrix[sorted(events["activity"].unique())].to_numpy(float).sum(axis=1)
    stranded = (event_cost > 1e-9) & (counted <= 1e-9)
    if stranded.any():
        raise ValueError(
            f"{int(stranded.sum())} intervals hold energy but no counted activity; "
            "energy and activity counts must be placed on the timeline the same way"
        )

    shape = sine_baseline(timeline)
    baseline = shape * 0.5 * sum(BACKGROUND_RANGE)
    noise = background_noise(shape, SIGNAL_SEED)
    signal = pd.DataFrame({
        "timestamp": timeline,
        "shape": shape,
        "baseline": baseline,
        "noise": noise,
        "event_cost": event_cost,
        "clean": baseline + event_cost,
        "signal": baseline + event_cost + noise,
    })
    return signal, matrix.reset_index()


def save_scope(scope: str, output: Path) -> None:
    """Save an auditable generated signal for inspection outside the model."""
    output.mkdir(parents=True, exist_ok=True)
    all_events = load_events()
    all_activities = sorted(all_events["activity"].unique())
    all_events = simulate(all_events, base_cost_table(all_activities))
    if scope == "learnable":
        events, audit = filter_learnable_events(all_events)
        audit.to_csv(output / "excluded_activities.csv", index=False)
    elif scope.startswith("top"):
        events = select_coverage(all_events, scope.removeprefix("top"))
    else:
        raise ValueError("scope must be topN or learnable")
    signal, matrix = build_signal_and_matrix(events)
    signal.to_csv(output / "signal.csv", index=False)
    matrix.to_csv(output / "event_matrix.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the controlled energy data")
    parser.add_argument("--scope", default="top3")
    parser.add_argument("--output", type=Path, default=ROOT / "generated_data")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    save_scope(arguments.scope, arguments.output)
