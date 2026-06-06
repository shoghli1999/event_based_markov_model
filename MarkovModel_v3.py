"""
MarkovModel_v3.py — three-variant event-based energy model (BPI Challenge 2019).

Pipeline (run via main()):
  1. Load XES log
  2. Find the top 3 most frequent variants
  3. Build EventObjects (tagged with variant_id, simulated durations)
  4. Generate synthetic signal from all three variants combined
  5. Build cost table and 30-min event matrix
  6. Build pooled + per-variant Markov chains and PNG plots
  7. Learn per-state energy; compare clean vs noisy vs true

Outputs → generated_signals_v3/
Uses fast vectorised event-cost binning (faster than v1/v2 nested loop).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pm4py


# separate RNG seed per variant so durations are reproducible but independent
VARIANT_RANDOM_SEEDS = {
    "variant_1": 42,
    "variant_2": 43,
    "variant_3": 44,
}

# base energy cost for every event type (my made-up "true" costs)
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


# one event = one activity in one case, plus a random duration and a variant label
@dataclass
class EventObject:
    e_id: int
    case_id: str
    event_type: str
    timestamp: pd.Timestamp
    duration_seconds: float
    case_history: List[int]
    variant_id: str

    def get_base_cost(self) -> float:
        # look up this event's base cost (0 if unknown)
        return BASE_COSTS.get(self.event_type, 0.0)


def load_xes_dataframe(xes_path: Path) -> pd.DataFrame:
    # read the xes log and turn it into a sorted pandas table
    log = pm4py.read_xes(str(xes_path))
    df = pm4py.convert_to_dataframe(log)
    required_columns = ["case:concept:name", "concept:name", "time:timestamp"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XES: {missing}")
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    return df


def find_top_k_variants(
    df: pd.DataFrame,
    k: int = 2,
) -> Tuple[List[Tuple[str, Tuple[str, ...], List[str], int]], pd.Series]:
    # build each case's event sequence, then pick the k most common variants
    case_sequences = (
        df.groupby("case:concept:name")["concept:name"].apply(lambda s: tuple(s.tolist()))
    )
    variant_counts = case_sequences.value_counts()
    if variant_counts.empty:
        raise ValueError("No variants found in the log.")
    if len(variant_counts) < k:
        raise ValueError(f"Requested {k} variants, but only {len(variant_counts)} exist.")

    # for each of the top k: keep its id, sequence, case ids and case count
    selected: List[Tuple[str, Tuple[str, ...], List[str], int]] = []
    for rank in range(k):
        variant_sequence = variant_counts.index[rank]
        case_count = int(variant_counts.iloc[rank])
        case_ids = case_sequences[case_sequences == variant_sequence].index.astype(str).tolist()
        selected.append((f"variant_{rank + 1}", variant_sequence, case_ids, case_count))
    return selected, variant_counts


def create_event_objects_for_cases(
    df: pd.DataFrame,
    case_ids: Iterable[str],
    variant_id: str,
    mean_duration_seconds: float = 180.0,
    std_duration_seconds: float = 20.0,
    random_seed: int = 42,
) -> List[EventObject]:
    # build one EventObject per row for these cases, tagged with the variant id
    rng = np.random.default_rng(random_seed)
    case_id_set = {str(c) for c in case_ids}
    filtered = df[df["case:concept:name"].astype(str).isin(case_id_set)].copy()
    filtered = filtered.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    events: List[EventObject] = []
    case_histories: Dict[str, List[int]] = {}

    for i, row in filtered.iterrows():
        case_id = str(row["case:concept:name"])
        case_history = case_histories.get(case_id, []).copy()
        # random duration around 180s, clipped to [120, 240]
        duration = float(rng.normal(mean_duration_seconds, std_duration_seconds))
        duration = float(np.clip(duration, 120.0, 240.0))

        event = EventObject(
            e_id=i,
            case_id=case_id,
            event_type=str(row["concept:name"]),
            timestamp=pd.Timestamp(row["time:timestamp"]),
            duration_seconds=duration,
            case_history=case_history,
            variant_id=variant_id,
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
    # the background daily sine wave, peaking at 2pm
    hours = timestamps.hour + timestamps.minute / 60.0
    phase = 2 * np.pi * (hours - peak_hour_shift) / 24.0
    return vertical_shift + amplitude * np.sin(phase)


def inject_event_costs(
    timeline: pd.DatetimeIndex,
    events: Sequence[EventObject],
    freq: str = "30min",
    event_cost_scale: float = 1.0,
    duration_scale: float = 0.01,
) -> np.ndarray:
    # fast version: bin each event into its 30-min slot and sum energies (no nested loop)
    if not events:
        return np.zeros(len(timeline), dtype=float)

    event_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([e.timestamp for e in events]),
            "cost": np.array(
                [
                    e.get_base_cost() * event_cost_scale + e.duration_seconds * duration_scale
                    for e in events
                ],
                dtype=float,
            ),
        }
    )
    event_df["interval_start"] = event_df["timestamp"].dt.floor(freq)
    grouped = event_df.groupby("interval_start", as_index=True)["cost"].sum()
    return grouped.reindex(timeline, fill_value=0.0).to_numpy(dtype=float)


def generate_three_variants_signal(
    events: Sequence[EventObject],
    output_dir: Path,
    freq: str = "30min",
) -> Tuple[Path, pd.DataFrame]:
    # build the signal from all three variants' events together and save it
    if not events:
        raise ValueError("No events provided for signal generation.")

    # timeline from first to last event in 30-min steps
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
    rng = np.random.default_rng(123)
    output_dir.mkdir(parents=True, exist_ok=True)
    # the three pieces: background wave, random noise, event spikes
    baseline = generate_sine_baseline(
        timeline,
        amplitude=params["amplitude"],
        vertical_shift=params["vertical_shift"],
    )
    noise = rng.normal(loc=0.0, scale=params["noise_std"], size=len(timeline))
    event_cost = inject_event_costs(
        timeline,
        events,
        freq=freq,
        event_cost_scale=params["event_cost_scale"],
    )
    total_energy = baseline + noise + event_cost

    # save both the clean version (no noise) and the noisy observed signal
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
    out_path = output_dir / "three_variants_signal.csv"
    df.to_csv(out_path, index=False)
    return out_path, df


def build_three_variants_cost_table(
    events: Sequence[EventObject],
    event_cost_scale: float = 1.0,
    duration_scale: float = 0.01,
) -> pd.DataFrame:
    # cost summary broken down per variant and per event type
    rows = []
    variant_ids = sorted({e.variant_id for e in events})
    event_types = sorted({e.event_type for e in events})
    for variant_id in variant_ids:
        for event_type in event_types:
            subset = [
                e for e in events if e.variant_id == variant_id and e.event_type == event_type
            ]
            if not subset:
                continue
            count = len(subset)
            base_cost = BASE_COSTS.get(event_type, 0.0)
            avg_duration = float(np.mean([e.duration_seconds for e in subset]))
            total_cost = sum(
                (base_cost * event_cost_scale) + (e.duration_seconds * duration_scale)
                for e in subset
            )
            rows.append(
                {
                    "variant_id": variant_id,
                    "event_type": event_type,
                    "count": count,
                    "base_cost": base_cost,
                    "avg_duration_seconds": round(avg_duration, 2),
                    "estimated_total_cost": round(float(total_cost), 2),
                }
            )

    table = pd.DataFrame(rows).sort_values("estimated_total_cost", ascending=False).reset_index(drop=True)
    return table


def build_three_variants_event_matrix(
    events: Sequence[EventObject],
    signal_df: pd.DataFrame,
    freq: str = "30min",
) -> pd.DataFrame:
    # grid of event counts per 30-min slot, across all three variants
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
    event_df["interval_start"] = event_df["timestamp"].dt.floor(freq)

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
    matrix["total_events"] = matrix.drop(columns=["timestamp"]).sum(axis=1)
    return matrix


def build_markov_chain(events: Sequence[EventObject]) -> pd.DataFrame:
    # count event-to-event transitions and turn them into probabilities
    event_df = pd.DataFrame(
        {
            "case_id": [e.case_id for e in events],
            "timestamp": [e.timestamp for e in events],
            "event_type": [e.event_type for e in events],
        }
    )
    event_df["timestamp"] = pd.to_datetime(event_df["timestamp"])
    event_df = event_df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    # next event per case; last event of a case goes to END
    event_df["next_state"] = event_df.groupby("case_id")["event_type"].shift(-1).fillna("END")

    transitions = (
        event_df.groupby(["event_type", "next_state"])
        .size()
        .reset_index(name="transition_count")
        .rename(columns={"event_type": "state"})
    )
    # probability = this transition's count / all transitions leaving that state
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
    # "guess and refine": start every energy at 1, update one state at a time, 20 passes
    # converges to the least-squares solution  ||X w - y||²
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

            # subtract what the other states already explain
            others = np.zeros_like(y, dtype=float)
            for k, other_state in enumerate(states):
                if k == j:
                    continue
                others += x[:, k] * estimates[other_state]

            # best value for this state given the leftover signal
            numer = float(np.dot(col, y - others))
            estimates[state] = numer / denom
    return estimates


def compute_true_state_energy_from_generator(
    events: Sequence[EventObject],
    states: Sequence[str],
    event_cost_scale: float = 1.0,
    duration_scale: float = 0.01,
) -> Dict[str, float]:
    # the real average energy per event type, used as ground truth for checking
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
    # subtract the baseline so only the event part remains to recover
    # clean target = event_cost only; noisy target = event_cost + noise
    event_only_clean = signal_df["without_noise"] - signal_df["baseline"]
    event_only_noisy = signal_df["with_noise"] - signal_df["baseline"]

    # recover energies from the clean signal and from the noisy signal
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
    # how much the noise moved the estimate
    result["noise_effect_on_learning"] = (
        result["learned_from_noisy_signal"] - result["learned_from_clean_signal"]
    )
    return result


def build_markov_learning_summary_table(learning_df: pd.DataFrame) -> pd.DataFrame:
    # tidy the comparison into a readable table and add absolute-error columns
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


def build_variants_overview_table(
    selected_variants: List[Tuple[str, Tuple[str, ...], List[str], int]],
) -> pd.DataFrame:
    # small table describing each selected variant (id, case count, sequence)
    rows = []
    for variant_id, sequence, case_ids, case_count in selected_variants:
        rows.append(
            {
                "variant_id": variant_id,
                "case_count": case_count,
                "sequence_length": len(sequence),
                "event_sequence": " -> ".join(sequence),
            }
        )
    return pd.DataFrame(rows)


def get_learning_states(event_matrix: pd.DataFrame) -> List[str]:
    # the states to learn = all matrix columns except the helper ones
    excluded = {"timestamp", "total_events"}
    return sorted(col for col in event_matrix.columns if col not in excluded)


def _node_label(name: str, max_len: int = 24) -> str:
    # shorten long event names so they fit inside the plotted nodes
    if len(name) <= max_len:
        return name
    return f"{name[: max_len - 3]}..."


def _linear_layout(nodes: Iterable[str], sequence_order: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    # place nodes left to right in the order of the variant's sequence (END last)
    rank = {state: idx for idx, state in enumerate(sequence_order)}
    rank["END"] = len(sequence_order)
    max_rank = max(rank.values()) if rank else 0
    pos: Dict[str, Tuple[float, float]] = {}
    extra_y = 0.0
    for node in nodes:
        if node in rank:
            pos[node] = (float(rank[node]), 0.0)
        else:
            # anything not in the main sequence is stacked off to the side
            pos[node] = (float(max_rank + 1), extra_y)
            extra_y += 0.35
    return pos


def plot_markov_chain(
    markov_df: pd.DataFrame,
    output_path: Path,
    title: str,
    sequence_order: Sequence[str] | None = None,
    min_probability: float = 0.001,
) -> Path:
    # draw the markov chain: events as nodes, transition probabilities on the arrows
    edges_df = markov_df[markov_df["transition_probability"] >= min_probability].copy()
    if edges_df.empty:
        raise ValueError(f"No transitions to plot for: {title}")

    # build a directed graph with the probability as each edge's label
    graph = nx.DiGraph()
    for _, row in edges_df.iterrows():
        probability = float(row["transition_probability"])
        graph.add_edge(
            str(row["state"]),
            str(row["next_state"]),
            weight=probability,
            label=f"{probability:.3f}",
        )

    # linear layout if i know the sequence, otherwise an automatic spring layout
    if sequence_order is not None:
        pos = _linear_layout(graph.nodes, sequence_order)
    else:
        pos = nx.spring_layout(graph, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(14, 7))
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=3200,
        node_color="#d9ecff",
        edgecolors="#1f4e79",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: _node_label(node) for node in graph.nodes},
        font_size=8,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=18,
        width=1.8,
        connectionstyle="arc3,rad=0.15",
    )
    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    # run the whole pipeline for the top 3 variants
    thesis_dir = Path(__file__).resolve().parent
    xes_path = thesis_dir / "BPI_Challenge_2019.xes"
    output_dir = thesis_dir / "generated_signals_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. load the log and pick the top 3 variants
    df = load_xes_dataframe(xes_path)
    selected_variants, variant_counts = find_top_k_variants(df, k=3)

    print("Three-variant mode: modeling top 3 most frequent variants.")
    print(f"Number of distinct variants in log: {len(variant_counts)}")
    for variant_id, sequence, case_ids, case_count in selected_variants:
        print(f"\n{variant_id} (cases: {case_count})")
        for step, event_name in enumerate(sequence, start=1):
            print(f"  {step:02d}. {event_name}")

    # 2. build events for each variant (different seed per variant)
    events: List[EventObject] = []
    for variant_id, _, case_ids, _ in selected_variants:
        variant_events = create_event_objects_for_cases(
            df,
            case_ids,
            variant_id=variant_id,
            random_seed=VARIANT_RANDOM_SEEDS.get(variant_id, 42),
        )
        events.extend(variant_events)

    print(
        f"\nBuilt EventObject list for all 3 variants: "
        f"{len(events)} events total (randomized durations per event)."
    )

    # 3. variant overview table
    overview_df = build_variants_overview_table(selected_variants)
    overview_path = output_dir / "three_variants_overview.csv"
    overview_df.to_csv(overview_path, index=False)
    print(f"\nSaved variant overview to: {overview_path}")

    # 4. build and save the synthetic signal
    written_file, signal_df = generate_three_variants_signal(events, output_dir=output_dir, freq="30min")
    print("\nGenerated synthetic signal for Variant 1 + Variant 2 + Variant 3 events:")
    print(f"- {written_file}")

    # 5. cost table (per variant, per event type)
    cost_table = build_three_variants_cost_table(events)
    cost_table_path = output_dir / "three_variants_cost_table.csv"
    cost_table.to_csv(cost_table_path, index=False)
    print("\nThree-variant cost table (sample):")
    print(cost_table.head(12).to_string(index=False))
    print(f"\nSaved full three-variant cost table to: {cost_table_path}")

    # 6. event matrix (counts per 30-min slot)
    event_matrix = build_three_variants_event_matrix(events, signal_df, freq="30min")
    event_matrix_path = output_dir / "three_variants_event_matrix_30min.csv"
    event_matrix.to_csv(event_matrix_path, index=False)
    print("\nSaved three-variant event matrix (30-minute intervals) to:")
    print(f"- {event_matrix_path}")

    # 7. pooled markov chain (all 3 variants mixed) + its plot
    pooled_markov_df = build_markov_chain(events)
    pooled_markov_path = output_dir / "three_variants_markov_chain_pooled.csv"
    pooled_markov_df.to_csv(pooled_markov_path, index=False)
    print("\nSaved pooled Markov chain (all 3 variants, data-driven probabilities) to:")
    print(f"- {pooled_markov_path}")

    # one markov chain + plot per variant (probabilities are ~1 inside a single variant)
    for variant_id, sequence, _, _ in selected_variants:
        variant_events = [e for e in events if e.variant_id == variant_id]
        variant_markov_df = build_markov_chain(variant_events)
        variant_markov_path = output_dir / f"{variant_id}_markov_chain.csv"
        variant_markov_df.to_csv(variant_markov_path, index=False)
        print(f"Saved {variant_id} Markov chain to: {variant_markov_path}")

        variant_plot_path = plot_markov_chain(
            variant_markov_df,
            output_dir / f"{variant_id}_markov_chain.png",
            title=f"{variant_id} Markov chain (transition probabilities on edges)",
            sequence_order=list(sequence),
        )
        print(f"Saved {variant_id} Markov chain plot to: {variant_plot_path}")

    # plot the pooled chain too (auto layout, shows branching probabilities < 1)
    pooled_plot_path = plot_markov_chain(
        pooled_markov_df,
        output_dir / "three_variants_markov_chain_pooled.png",
        title="Pooled three-variant Markov chain (transition probabilities on edges)",
        sequence_order=None,
    )
    print(f"Saved pooled Markov chain plot to: {pooled_plot_path}")

    # 8. recover the energies from the signal and compare to the true values
    learning_states = get_learning_states(event_matrix)
    learning_df = evaluate_markov_state_learning(
        signal_df=signal_df,
        event_matrix=event_matrix,
        states=learning_states,
    )
    true_state_energy = compute_true_state_energy_from_generator(events, learning_states)
    learning_df["true_generator_state_energy"] = learning_df["state"].map(true_state_energy)
    # error = recovered - true, for clean and noisy
    learning_df["error_vs_true_clean"] = (
        learning_df["learned_from_clean_signal"] - learning_df["true_generator_state_energy"]
    )
    learning_df["error_vs_true_noisy"] = (
        learning_df["learned_from_noisy_signal"] - learning_df["true_generator_state_energy"]
    )

    learning_path = output_dir / "three_variants_markov_state_learning.csv"
    learning_df.to_csv(learning_path, index=False)
    print("\nMarkov state-energy learning summary (all states used in matrix):")
    print(learning_df.to_string(index=False))
    print(f"\nSaved state-learning comparison to: {learning_path}")

    # 9. save the readable summary table
    learning_summary_df = build_markov_learning_summary_table(learning_df)
    learning_summary_path = output_dir / "three_variants_markov_state_learning_summary.csv"
    learning_summary_df.to_csv(learning_summary_path, index=False)
    print(f"Saved readable learning summary table to: {learning_summary_path}")


if __name__ == "__main__":
    main()
