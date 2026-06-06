"""
MarkovModel.py — single-variant event-based energy model (BPI Challenge 2019).

Pipeline (run via main()):
  1. Load XES log
  2. Find the most frequent process variant (Variant 1)
  3. Build EventObjects with simulated durations
  4. Generate synthetic energy signal (sine baseline + noise + event costs)
  5. Build cost table and 30-min event occurrence matrix
  6. Build Markov chain (empirical transition probabilities)
  7. Learn per-state energy from the signal; compare clean vs noisy vs true
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pm4py


# base energy cost for every event type. these numbers are made up by me and act as
# the "true" costs that i hide inside the signal and later try to recover.
BASE_COSTS: Dict[str, float] = {
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
}


# one event object = one activity in one case (one row of the log),
# plus a random duration that i add myself.
@dataclass
class EventObject:
    e_id: int
    case_id: str
    event_type: str
    timestamp: pd.Timestamp
    duration_seconds: float
    case_history: List[int]

    def get_base_cost(self) -> float:
        # look up this event's base cost in the table above (0 if unknown)
        return BASE_COSTS.get(self.event_type, 0.0)


def load_xes_dataframe(xes_path: Path) -> pd.DataFrame:
    # read the xes log and turn it into a pandas table
    log = pm4py.read_xes(str(xes_path))
    df = pm4py.convert_to_dataframe(log)
    # check the three columns i rely on are present
    required_columns = ["case:concept:name", "concept:name", "time:timestamp"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XES: {missing}")
    # parse timestamps and sort each case by time so the events are in order
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    return df


def find_most_frequent_variant(
    df: pd.DataFrame,
) -> Tuple[Tuple[str, ...], List[str], pd.Series]:
    # a variant is the exact sequence of events a case went through.
    # here i build that sequence per case and pick the single most common one.
    case_sequences = (
        df.groupby("case:concept:name")["concept:name"].apply(lambda s: tuple(s.tolist()))
    )
    variant_counts = case_sequences.value_counts()
    if variant_counts.empty:
        raise ValueError("No variants found in the log.")
    # the most frequent sequence, and the ids of all cases that follow it
    top_variant = variant_counts.index[0]
    top_cases = case_sequences[case_sequences == top_variant].index.astype(str).tolist()
    return top_variant, top_cases, variant_counts


def create_event_objects_for_cases(
    df: pd.DataFrame,
    case_ids: Iterable[str],
    mean_duration_seconds: float = 180.0,
    std_duration_seconds: float = 20.0,
    random_seed: int = 42,
) -> List[EventObject]:
    # build one EventObject per log row for the chosen cases.
    # the seed makes the random durations the same every run.
    rng = np.random.default_rng(random_seed)
    case_id_set = {str(c) for c in case_ids}
    # keep only the rows that belong to the chosen cases, sorted by time
    filtered = df[df["case:concept:name"].astype(str).isin(case_id_set)].copy()
    filtered = filtered.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    events: List[EventObject] = []
    case_histories: Dict[str, List[int]] = {}

    for i, row in filtered.iterrows():
        case_id = str(row["case:concept:name"])
        # remember which events happened before in this case
        case_history = case_histories.get(case_id, []).copy()
        # give this event a random duration around 180s, clipped to [120, 240]
        duration = float(rng.normal(mean_duration_seconds, std_duration_seconds))
        duration = float(np.clip(duration, 120.0, 240.0))

        event = EventObject(
            e_id=i,
            case_id=case_id,
            event_type=str(row["concept:name"]),
            timestamp=pd.Timestamp(row["time:timestamp"]),
            duration_seconds=duration,
            case_history=case_history,
        )
        events.append(event)
        case_histories.setdefault(case_id, []).append(i)
    return events


def generate_sine_baseline(
    timestamps: pd.DatetimeIndex,
    amplitude: float = 2.0,
    vertical_shift: float = 5.0,
    peak_hour_shift: float = 14.0,
) -> np.ndarray:
    # the background load: a smooth daily sine wave that peaks in the afternoon (2pm)
    hours = timestamps.hour + timestamps.minute / 60.0
    phase = 2 * np.pi * (hours - peak_hour_shift) / 24.0
    return vertical_shift + amplitude * np.sin(phase)


def inject_event_costs(
    timeline: pd.DatetimeIndex,
    events: Sequence[EventObject],
    event_cost_scale: float = 1.0,
    duration_scale: float = 0.01,
) -> np.ndarray:
    # for every 30-min slot, add up the energy of all events that fall inside it.
    # energy of one event = base cost + a small term for its duration.
    event_cost = np.zeros(len(timeline), dtype=float)
    interval_start = pd.Series(timeline)
    # the end of each slot is the start of the next one
    interval_end = interval_start.shift(-1)
    interval_end.iloc[-1] = interval_start.iloc[-1] + (timeline[1] - timeline[0])

    for idx, start_ts in enumerate(interval_start):
        end_ts = interval_end.iloc[idx]
        total_cost = 0.0
        # slow way: check every event against this slot (fine for one variant)
        for event in events:
            if start_ts <= event.timestamp < end_ts:
                total_cost += (
                    event.get_base_cost() * event_cost_scale
                    + event.duration_seconds * duration_scale
                )
        event_cost[idx] = total_cost
    return event_cost


def generate_variant1_signal(
    events: Sequence[EventObject],
    output_dir: Path,
    freq: str = "30min",
) -> Tuple[Path, pd.DataFrame]:
    # build the full synthetic signal for variant 1 and save it as csv
    if not events:
        raise ValueError("No events provided for signal generation.")

    # the timeline runs from the first event to the last, in 30-min steps
    min_ts = min(e.timestamp for e in events).floor(freq)
    max_ts = max(e.timestamp for e in events).ceil(freq)
    timeline = pd.date_range(start=min_ts, end=max_ts, freq=freq)
    if len(timeline) < 2:
        timeline = pd.date_range(start=min_ts, periods=2, freq=freq)

    params = {
        "amplitude": 2.0,
        "vertical_shift": 5.0,
        "noise_std": 0.5,
        "event_cost_scale": 1.0,
    }
    # fixed seed so the noise is reproducible
    rng = np.random.default_rng(123)
    output_dir.mkdir(parents=True, exist_ok=True)
    # the three pieces of the signal: background wave, random noise, event spikes
    baseline = generate_sine_baseline(
        timeline,
        amplitude=params["amplitude"],
        vertical_shift=params["vertical_shift"],
    )
    noise = rng.normal(loc=0.0, scale=params["noise_std"], size=len(timeline))
    event_cost = inject_event_costs(
        timeline,
        events,
        event_cost_scale=params["event_cost_scale"],
    )
    # the observed signal is the sum of all three
    total_energy = baseline + noise + event_cost

    # save both the clean version (no noise) and the noisy one
    df = pd.DataFrame(
        {
            "timestamp": timeline,
            "baseline": baseline,
            "noise": noise,
            "event_cost": event_cost,
            "without_noise": baseline + event_cost,
            "with_noise": total_energy,
            "total": total_energy,
        }
    )
    out_path = output_dir / "variant1_signal.csv"
    df.to_csv(out_path, index=False)
    return out_path, df


def build_variant1_cost_table(
    events: Sequence[EventObject],
    event_cost_scale: float = 1.0,
    duration_scale: float = 0.01,
) -> pd.DataFrame:
    # readable summary: per event type, how many happened and their total energy
    rows = []
    event_types = sorted({e.event_type for e in events})
    for event_type in event_types:
        subset = [e for e in events if e.event_type == event_type]
        count = len(subset)
        base_cost = BASE_COSTS.get(event_type, 0.0)
        avg_duration = float(np.mean([e.duration_seconds for e in subset])) if subset else 0.0
        total_cost = sum(
            (base_cost * event_cost_scale) + (e.duration_seconds * duration_scale)
            for e in subset
        )
        rows.append(
            {
                "event_type": event_type,
                "count": count,
                "base_cost": base_cost,
                "avg_duration_seconds": round(avg_duration, 2),
                "estimated_total_cost": round(float(total_cost), 2),
            }
        )

    # sort so the most expensive event types are on top
    table = pd.DataFrame(rows).sort_values("estimated_total_cost", ascending=False).reset_index(drop=True)
    return table


def build_variant1_event_matrix(
    events: Sequence[EventObject],
    signal_df: pd.DataFrame,
    freq: str = "30min",
) -> pd.DataFrame:
    # grid: rows = 30-min slots, columns = event types, cell = how many happened
    if not events:
        raise ValueError("No events provided for event matrix generation.")

    timeline = pd.to_datetime(signal_df["timestamp"]).sort_values().drop_duplicates()
    event_df = pd.DataFrame(
        {
            "timestamp": [e.timestamp for e in events],
            "event_type": [e.event_type for e in events],
        }
    )
    event_df["timestamp"] = pd.to_datetime(event_df["timestamp"])
    # round each event down to the start of its 30-min slot
    event_df["interval_start"] = event_df["timestamp"].dt.floor(freq)

    # count events per (slot, type) and reshape into the grid, zero-filled
    matrix = (
        event_df.groupby(["interval_start", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(timeline, fill_value=0)
        .sort_index()
        .astype(int)
    )
    matrix.index.name = "timestamp"
    matrix = matrix.reset_index()
    # extra column with the total number of events in each slot
    matrix["total_events"] = matrix.drop(columns=["timestamp"]).sum(axis=1)
    return matrix


def build_variant1_markov_chain(events: Sequence[EventObject]) -> pd.DataFrame:
    # the markov chain: count how often one event is followed by another,
    # then turn those counts into probabilities.
    event_df = pd.DataFrame(
        {
            "case_id": [e.case_id for e in events],
            "timestamp": [e.timestamp for e in events],
            "event_type": [e.event_type for e in events],
        }
    )
    event_df["timestamp"] = pd.to_datetime(event_df["timestamp"])
    event_df = event_df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    # next event in the same case; the last event of a case points to END
    event_df["next_state"] = event_df.groupby("case_id")["event_type"].shift(-1).fillna("END")

    # count each (state -> next_state) pair
    transitions = (
        event_df.groupby(["event_type", "next_state"])
        .size()
        .reset_index(name="transition_count")
        .rename(columns={"event_type": "state"})
    )
    # probability = count of this transition / all transitions leaving that state
    totals = transitions.groupby("state")["transition_count"].transform("sum")
    transitions["transition_probability"] = transitions["transition_count"] / totals
    transitions = transitions.sort_values(["state", "next_state"]).reset_index(drop=True)
    return transitions[["state", "next_state", "transition_count", "transition_probability"]]


def estimate_state_energy_iterative(
    event_matrix: pd.DataFrame,
    target_series: pd.Series,
    states: Sequence[str],
    iterations: int = 20,
) -> Dict[str, float]:
    # Simple "guess then average" learning by coordinate updates.
    # This converges to a least-squares estimate for state energy.
    # i start every state's energy at 1, then refine one state at a time, 20 times.
    estimates = {state: 1.0 for state in states}
    x = event_matrix[list(states)].to_numpy(dtype=float)
    y = target_series.to_numpy(dtype=float)

    for _ in range(iterations):
        for j, state in enumerate(states):
            col = x[:, j]
            denom = float(np.dot(col, col))
            if denom == 0.0:
                estimates[state] = 0.0
                continue

            # what the other states already explain in the signal
            others = np.zeros_like(y, dtype=float)
            for k, other_state in enumerate(states):
                if k == j:
                    continue
                others += x[:, k] * estimates[other_state]

            # best value for this state given everything that's left over
            numer = float(np.dot(col, y - others))
            estimates[state] = numer / denom
    return estimates


def compute_true_state_energy_from_generator(
    events: Sequence[EventObject],
    states: Sequence[str],
    event_cost_scale: float = 1.0,
    duration_scale: float = 0.01,
) -> Dict[str, float]:
    # the real (average) energy per event type, used as ground truth to check the recovery
    true_costs: Dict[str, float] = {}
    for state in states:
        state_events = [e for e in events if e.event_type == state]
        if not state_events:
            true_costs[state] = 0.0
            continue
        per_occurrence = [
            (e.get_base_cost() * event_cost_scale) + (e.duration_seconds * duration_scale)
            for e in state_events
        ]
        true_costs[state] = float(np.mean(per_occurrence))
    return true_costs


def evaluate_markov_state_learning(
    signal_df: pd.DataFrame,
    event_matrix: pd.DataFrame,
    states: Sequence[str],
) -> pd.DataFrame:
    # Event-only targets from signal:
    # clean target: exactly what generator used for event contribution
    # noisy target: includes noise; if noise cancels, it should approach clean estimates
    # i subtract the baseline so only the event part is left to recover
    event_only_clean = signal_df["without_noise"] - signal_df["baseline"]
    event_only_noisy = signal_df["with_noise"] - signal_df["baseline"]

    # run the recovery on the clean signal and again on the noisy signal
    learned_clean = estimate_state_energy_iterative(
        event_matrix=event_matrix,
        target_series=event_only_clean,
        states=states,
    )
    learned_noisy = estimate_state_energy_iterative(
        event_matrix=event_matrix,
        target_series=event_only_noisy,
        states=states,
    )

    result = pd.DataFrame(
        {
            "state": list(states),
            "learned_from_clean_signal": [learned_clean[s] for s in states],
            "learned_from_noisy_signal": [learned_noisy[s] for s in states],
        }
    )
    # how much the noise shifted the estimate
    result["noise_effect_on_learning"] = (
        result["learned_from_noisy_signal"] - result["learned_from_clean_signal"]
    )
    return result


def build_markov_learning_summary_table(learning_df: pd.DataFrame) -> pd.DataFrame:
    # tidy the results into a readable table and add absolute-error columns
    summary = learning_df.copy()
    summary["abs_error_clean"] = summary["error_vs_true_clean"].abs()
    summary["abs_error_noisy"] = summary["error_vs_true_noisy"].abs()
    summary["abs_noise_effect"] = summary["noise_effect_on_learning"].abs()

    ordered_columns = [
        "state",
        "true_generator_state_energy",
        "learned_from_clean_signal",
        "learned_from_noisy_signal",
        "noise_effect_on_learning",
        "error_vs_true_clean",
        "error_vs_true_noisy",
        "abs_error_clean",
        "abs_error_noisy",
        "abs_noise_effect",
    ]
    summary = summary[ordered_columns]

    numeric_cols = [col for col in summary.columns if col != "state"]
    summary[numeric_cols] = summary[numeric_cols].round(6)

    # add an AVERAGE row at the bottom
    avg_row = {"state": "AVERAGE"}
    for col in numeric_cols:
        avg_row[col] = round(float(summary[col].mean()), 6)
    summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)
    return summary


def main() -> None:
    # run the whole pipeline for variant 1, step by step
    thesis_dir = Path(__file__).resolve().parent
    xes_path = thesis_dir / "BPI_Challenge_2019.xes"
    output_dir = thesis_dir / "generated_signals"

    # 1. load the log and find the most frequent variant
    df = load_xes_dataframe(xes_path)
    top_variant, top_cases, variant_counts = find_most_frequent_variant(df)

    print("Variant 1 selected (most frequent variant).")
    print(f"Number of distinct variants: {len(variant_counts)}")
    print(f"Variant 1 frequency (number of cases): {variant_counts.iloc[0]}")
    print("\nVariant 1 event sequence:")
    for step, event_name in enumerate(top_variant, start=1):
        print(f"{step:02d}. {event_name}")

    # 2. build event objects (with random durations) for that variant
    events = create_event_objects_for_cases(df, top_cases)
    print(
        f"\nBuilt Variant 1 EventObject list for {len(top_cases)} cases and {len(events)} events "
        "(all with randomized durations)."
    )

    # 3. build and save the synthetic signal
    written_file, signal_df = generate_variant1_signal(events, output_dir=output_dir, freq="30min")
    print("\nGenerated synthetic signal for Variant 1 events only:")
    print(f"- {written_file}")

    # 4. cost table
    variant1_cost_table = build_variant1_cost_table(events)
    cost_table_path = output_dir / "variant1_cost_table.csv"
    variant1_cost_table.to_csv(cost_table_path, index=False)
    print("\nVariant 1 cost table (top rows):")
    print(variant1_cost_table.to_string(index=False))
    print(f"\nSaved full Variant 1 cost table to: {cost_table_path}")

    # 5. event matrix (counts per 30-min slot)
    event_matrix = build_variant1_event_matrix(events, signal_df, freq="30min")
    event_matrix_path = output_dir / "variant1_event_matrix_30min.csv"
    event_matrix.to_csv(event_matrix_path, index=False)
    print("\nSaved Variant 1 event matrix (30-minute intervals, full signal range) to:")
    print(f"- {event_matrix_path}")

    # 6. markov chain (transition probabilities)
    variant1_states = list(top_variant)
    markov_chain_df = build_variant1_markov_chain(events)
    markov_chain_path = output_dir / "variant1_markov_chain.csv"
    markov_chain_df.to_csv(markov_chain_path, index=False)
    print("\nSaved Variant 1 Markov chain transitions to:")
    print(f"- {markov_chain_path}")

    # 7. recover the energies from the signal and compare to the true values
    learning_df = evaluate_markov_state_learning(
        signal_df=signal_df,
        event_matrix=event_matrix,
        states=variant1_states,
    )
    true_state_energy = compute_true_state_energy_from_generator(events, variant1_states)
    learning_df["true_generator_state_energy"] = learning_df["state"].map(true_state_energy)
    # error = recovered - true, for both clean and noisy signals
    learning_df["error_vs_true_clean"] = (
        learning_df["learned_from_clean_signal"] - learning_df["true_generator_state_energy"]
    )
    learning_df["error_vs_true_noisy"] = (
        learning_df["learned_from_noisy_signal"] - learning_df["true_generator_state_energy"]
    )

    learning_path = output_dir / "variant1_markov_state_learning.csv"
    learning_df.to_csv(learning_path, index=False)
    print("\nMarkov state-energy learning summary:")
    print(learning_df.to_string(index=False))
    print(f"\nSaved state-learning comparison to: {learning_path}")

    # 8. save the readable summary table
    learning_summary_df = build_markov_learning_summary_table(learning_df)
    learning_summary_path = output_dir / "variant1_markov_state_learning_summary.csv"
    learning_summary_df.to_csv(learning_summary_path, index=False)
    print(f"Saved readable learning summary table to: {learning_summary_path}")


if __name__ == "__main__":
    main()
