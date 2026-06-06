"""
MarkovModel_clean.py
────────────────────
Event-based Markov energy model — BPI Challenge 2019.

Model
─────
  • Each event type = one Markov state.
  • Entering a state consumes energy; the transition itself carries no cost.
  • Within a single variant the chain is deterministic (all probabilities ≈ 1).
  • When multiple variants are pooled, probabilities reflect the empirical mixture.

Learning
────────
  • A synthetic signal is built:  baseline (sine)  +  Gaussian noise  +  per-event costs.
  • State energies are recovered by coordinate descent on the event-occurrence matrix
    (the "guess and refine" loop described in the thesis).
  • Results are compared to true generator values to verify noise averages out.

Usage
─────
  python MarkovModel_clean.py           # top-3 variants  (default)
  python MarkovModel_clean.py --k 1     # variant 1 only
  python MarkovModel_clean.py --k 2     # variants 1–2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pm4py


# ─── Configuration ────────────────────────────────────────────────────────────

FREQ           = "30min"    # signal / matrix resolution
SINE_AMPLITUDE = 2.0        # sine baseline amplitude
SINE_SHIFT     = 5.0        # sine baseline vertical offset
SINE_PEAK_HOUR = 14.0       # hour of peak energy in the day
NOISE_STD      = 0.5        # Gaussian noise standard deviation
SIGNAL_SEED    = 123        # RNG seed for noise (reproducible)
DURATION_MEAN  = 180.0      # mean event duration  (seconds)
DURATION_STD   = 20.0       # std  event duration  (seconds)
DURATION_MIN   = 120.0      # lower clip
DURATION_MAX   = 240.0      # upper clip
DURATION_SCALE = 0.01       # energy = BASE_COST + duration × DURATION_SCALE
LEARN_ITERS    = 20         # coordinate-descent iterations

BASE_COSTS: Dict[str, float] = {
    "Record Goods Receipt":               99,
    "Create Purchase Order Item":         98,
    "Record Invoice Receipt":             97,
    "Vendor creates invoice":             54,
    "Clear Invoice":                      56,
    "Record Service Entry Sheet":         81,
    "Remove Payment Block":               59,
    "Create Purchase Requisition Item":   11,
    "Receive Order Confirmation":         66,
    "Change Quantity":                     2,
    "Change Price":                        1,
    "Delete Purchase Order item":          8,
    "Cancel Invoice Receipt":              9,
    "Change Approval for Purchase Order":  21,
    "Vendor creates debit memo":          20,
    "Change Delivery Indicator":          52,
    "Cancel Goods Receipt":               10,
    "Release Purchase Order":             60,
    "SRM: In Transfer to Execution Syst.": 87,
    "SRM: Created":                       78,
    "SRM: Complete":                      77,
    "SRM: Awaiting Approval":             63,
    "SRM Document Completed":             70,
    "SRM: Ordered":                       44,
    "SRM Change was Transmitted":         92,
    "Reactivate Purchase Order Item":     57,
    "Block Purchase Order Item":          31,
    "Cancel Subsequent Invoice":          32,
    "Change Storage Location":            36,
    "Update Order Confirmation":          22,
    "Record Subsequent Invoice":          40,
    "Release Purchase Requisition":       62,
    "Set Payment Block":                  26,
    "SRM: Deleted":                        5,
    "Change Currency":                   150,
    "Change Final Invoice Indicator":    144,
    "SRM: Transaction Completed":        121,
    "SRM: Incomplete":                   111,
    "SRM: Held":                          88,
    "Change payment term":                 4,
    "Change Rejection Indicator":        200,
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Variant:
    variant_id: str
    sequence: Tuple[str, ...]   # ordered event types for this variant
    case_ids: List[str]
    case_count: int


@dataclass
class Event:
    case_id: str
    event_type: str
    timestamp: pd.Timestamp
    duration_seconds: float
    variant_id: str

    def energy(self) -> float:
        """Energy consumed when this state is entered (base cost + duration term)."""
        return BASE_COSTS.get(self.event_type, 0.0) + self.duration_seconds * DURATION_SCALE


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_xes(xes_path: Path) -> pd.DataFrame:
    log = pm4py.read_xes(str(xes_path))
    df  = pm4py.convert_to_dataframe(log)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    return df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)


def find_top_variants(df: pd.DataFrame, k: int) -> List[Variant]:
    """Return the k most frequent process variants ranked by case count."""
    sequences = df.groupby("case:concept:name")["concept:name"].apply(tuple)
    counts    = sequences.value_counts()
    if len(counts) < k:
        raise ValueError(f"Log has only {len(counts)} distinct variants; cannot select {k}.")

    variants = []
    for rank in range(k):
        seq      = counts.index[rank]
        case_ids = sequences[sequences == seq].index.astype(str).tolist()
        variants.append(Variant(
            variant_id = f"variant_{rank + 1}",
            sequence   = seq,
            case_ids   = case_ids,
            case_count = int(counts.iloc[rank]),
        ))
    return variants


def build_events(df: pd.DataFrame, variants: List[Variant]) -> List[Event]:
    """Create one Event per log row for every case in the selected variants.
    Each variant gets its own RNG seed so durations are reproducible and independent."""
    events: List[Event] = []
    for idx, v in enumerate(variants):
        rng      = np.random.default_rng(42 + idx)
        filtered = (
            df[df["case:concept:name"].astype(str).isin(set(v.case_ids))]
            .sort_values(["case:concept:name", "time:timestamp"])
            .reset_index(drop=True)
        )
        for _, row in filtered.iterrows():
            duration = float(np.clip(
                rng.normal(DURATION_MEAN, DURATION_STD), DURATION_MIN, DURATION_MAX
            ))
            events.append(Event(
                case_id          = str(row["case:concept:name"]),
                event_type       = str(row["concept:name"]),
                timestamp        = pd.Timestamp(row["time:timestamp"]),
                duration_seconds = duration,
                variant_id       = v.variant_id,
            ))
    return events


# ─── Synthetic signal generation ──────────────────────────────────────────────

def _sine_baseline(timeline: pd.DatetimeIndex) -> np.ndarray:
    """Daily sine-wave background load."""
    hours = timeline.hour + timeline.minute / 60.0
    return SINE_SHIFT + SINE_AMPLITUDE * np.sin(
        2 * np.pi * (hours - SINE_PEAK_HOUR) / 24.0
    )


def _event_costs_binned(events: Sequence[Event], timeline: pd.DatetimeIndex) -> np.ndarray:
    """Sum per-event energy into 30-min buckets aligned to the signal timeline.
    Fully vectorised — no Python loops over intervals."""
    df = pd.DataFrame({
        "interval": pd.DatetimeIndex([e.timestamp for e in events]).floor(FREQ),
        "cost":     [e.energy() for e in events],
    })
    return (
        df.groupby("interval")["cost"]
        .sum()
        .reindex(timeline, fill_value=0.0)
        .to_numpy(dtype=float)
    )


def generate_signal(events: Sequence[Event], out_dir: Path) -> Tuple[Path, pd.DataFrame]:
    """Build the synthetic energy signal and save it as CSV.

    Columns:
      baseline   – sine wave (background load)
      noise      – Gaussian random noise
      event_cost – binned energy from process events
      clean      – baseline + event_cost        (no noise; used as learning target)
      signal     – baseline + event_cost + noise (observed signal)
    """
    min_ts   = min(e.timestamp for e in events).floor(FREQ)
    max_ts   = max(e.timestamp for e in events).ceil(FREQ)
    timeline = pd.date_range(min_ts, max_ts, freq=FREQ)

    rng        = np.random.default_rng(SIGNAL_SEED)
    baseline   = _sine_baseline(timeline)
    noise      = rng.normal(0.0, NOISE_STD, len(timeline))
    event_cost = _event_costs_binned(events, timeline)

    df = pd.DataFrame({
        "timestamp":  timeline,
        "baseline":   baseline,
        "noise":      noise,
        "event_cost": event_cost,
        "clean":      baseline + event_cost,
        "signal":     baseline + noise + event_cost,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "signal.csv"
    df.to_csv(path, index=False)
    return path, df


# ─── Event occurrence matrix ──────────────────────────────────────────────────

def build_event_matrix(events: Sequence[Event], signal_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = 30-min intervals, columns = event types, values = count of occurrences.
    Aligned to the full signal timeline (zero-filled for empty intervals)."""
    timeline = pd.DatetimeIndex(signal_df["timestamp"])
    df = pd.DataFrame({
        "interval":   pd.DatetimeIndex([e.timestamp for e in events]).floor(FREQ),
        "event_type": [e.event_type for e in events],
    })
    matrix = (
        df.groupby(["interval", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(timeline, fill_value=0)
        .astype(int)
    )
    matrix.index.name = "timestamp"
    return matrix.reset_index()


# ─── Markov chain ─────────────────────────────────────────────────────────────

def build_markov_chain(events: Sequence[Event]) -> pd.DataFrame:
    """Empirical transition probability table.

    For a single deterministic variant → all probabilities are 1.0.
    For pooled / mixed variants → probabilities reflect the empirical branching.
    Probabilities are always data-driven, never hardcoded.
    """
    df = pd.DataFrame({
        "case_id":   [e.case_id for e in events],
        "timestamp": pd.to_datetime([e.timestamp for e in events]),
        "state":     [e.event_type for e in events],
    }).sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    # Last event in each case transitions to END
    df["next_state"] = df.groupby("case_id")["state"].shift(-1).fillna("END")

    counts = df.groupby(["state", "next_state"]).size().reset_index(name="count")
    counts["probability"] = (
        counts["count"] / counts.groupby("state")["count"].transform("sum")
    )
    return counts.sort_values(["state", "next_state"]).reset_index(drop=True)


# ─── Energy learning (coordinate descent) ────────────────────────────────────

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    states: List[str],
    iterations: int = LEARN_ITERS,
) -> Dict[str, float]:
    """Recover per-state energy by iterative coordinate updates ("guess and refine").

    For each state j in each pass:
        w_j  ←  col_j · (y − Σ_{k≠j} col_k × w_k)  /  (col_j · col_j)

    This is coordinate descent on the least-squares objective  ||X w − y||²
    and converges to the OLS solution.  With a large number of intervals and
    zero-mean noise the result closely matches the true generator energies.
    """
    w = {s: 1.0 for s in states}
    for _ in range(iterations):
        for j, s in enumerate(states):
            col   = X[:, j]
            denom = float(np.dot(col, col))
            if denom == 0.0:
                w[s] = 0.0
                continue
            # Residual after removing all other states' contributions
            residual = y.copy()
            for k, other in enumerate(states):
                if k != j:
                    residual -= X[:, k] * w[other]
            w[s] = float(np.dot(col, residual) / denom)
    return w


def build_learning_summary(
    signal_df: pd.DataFrame,
    matrix: pd.DataFrame,
    events: Sequence[Event],
    states: List[str],
) -> pd.DataFrame:
    """Learn state energies from the clean and noisy signal, then compare to truth.

    Targets (baseline already subtracted so only the event-cost part is modelled):
      clean  →  event_cost exactly       (no noise contamination)
      noisy  →  event_cost + noise       (tests whether noise averages out)
    """
    X       = matrix[states].to_numpy(dtype=float)
    clean_y = (signal_df["clean"]  - signal_df["baseline"]).to_numpy(dtype=float)
    noisy_y = (signal_df["signal"] - signal_df["baseline"]).to_numpy(dtype=float)

    w_clean = _coordinate_descent(X, clean_y, states)
    w_noisy = _coordinate_descent(X, noisy_y, states)

    # Ground truth: average (base_cost + duration × scale) over all events of each type
    true_e = {
        s: float(np.mean([e.energy() for e in events if e.event_type == s]))
        for s in states
    }

    rows = []
    for s in sorted(states):
        lc, ln, tr = w_clean[s], w_noisy[s], true_e[s]
        rows.append({
            "state":          s,
            "true_energy":    round(tr,       6),
            "learned_clean":  round(lc,       6),
            "learned_noisy":  round(ln,       6),
            "error_clean":    round(lc - tr,  6),   # positive = overestimate
            "error_noisy":    round(ln - tr,  6),
            "noise_effect":   round(ln - lc,  6),   # how much noise shifted the estimate
        })

    summary = pd.DataFrame(rows)
    # Append a convenience AVERAGE row
    avg = {"state": "AVERAGE"}
    for col in summary.columns[1:]:
        avg[col] = round(float(summary[col].mean()), 6)
    return pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)


# ─── Reporting helpers ────────────────────────────────────────────────────────

def build_variant_overview(variants: List[Variant]) -> pd.DataFrame:
    return pd.DataFrame([{
        "variant_id":      v.variant_id,
        "case_count":      v.case_count,
        "sequence_length": len(v.sequence),
        "event_sequence":  " → ".join(v.sequence),
    } for v in variants])


def build_cost_table(events: Sequence[Event]) -> pd.DataFrame:
    """Per-variant, per-event-type cost breakdown."""
    rows = []
    for vid in sorted({e.variant_id for e in events}):
        for etype in sorted({e.event_type for e in events}):
            subset = [e for e in events if e.variant_id == vid and e.event_type == etype]
            if not subset:
                continue
            rows.append({
                "variant_id":           vid,
                "event_type":           etype,
                "count":                len(subset),
                "base_cost":            BASE_COSTS.get(etype, 0.0),
                "avg_duration_seconds": round(float(np.mean([e.duration_seconds for e in subset])), 2),
                "estimated_total_cost": round(sum(e.energy() for e in subset), 2),
            })
    return (pd.DataFrame(rows)
              .sort_values("estimated_total_cost", ascending=False)
              .reset_index(drop=True))


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _short_label(name: str, max_len: int = 22) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def _linear_pos(
    nodes: Sequence[str], sequence: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    """Left-to-right layout following the variant's event order."""
    rank = {s: i for i, s in enumerate(sequence)}
    rank["END"] = len(sequence)
    top = max(rank.values())
    pos: Dict[str, Tuple[float, float]] = {}
    extra = 0.0
    for n in nodes:
        if n in rank:
            pos[n] = (float(rank[n]), 0.0)
        else:                              # nodes not in the main sequence
            pos[n] = (float(top + 1), extra)
            extra  += 0.4
    return pos


def plot_markov_chain(
    chain_df: pd.DataFrame,
    title: str,
    output_path: Path,
    sequence_order: Optional[Sequence[str]] = None,
) -> None:
    """Draw states as nodes and transition probabilities as edge labels."""
    edges = chain_df[chain_df["probability"] >= 0.001]
    if edges.empty:
        return

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(str(row["state"]), str(row["next_state"]),
                   label=f"{row['probability']:.3f}")

    pos = (_linear_pos(list(G.nodes), sequence_order)
           if sequence_order is not None
           else nx.spring_layout(G, seed=42, k=2.0))

    fig, ax = plt.subplots(figsize=(14, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000,
                           node_color="#d9ecff", edgecolors="#1f4e79", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            labels={n: _short_label(n) for n in G.nodes})
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18,
                           width=1.8, connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "label"),
                                 font_size=8, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Plot saved → {output_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(k: int = 3) -> None:
    thesis_dir = Path(__file__).resolve().parent
    xes_path   = thesis_dir / "BPI_Challenge_2019.xes"
    out_dir    = thesis_dir / f"generated_signals_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load log ───────────────────────────────────────────────────────────
    print("Loading XES log …")
    df = load_xes(xes_path)

    # ── 2. Select top-k variants ──────────────────────────────────────────────
    variants = find_top_variants(df, k)
    total_distinct = df.groupby("case:concept:name")["concept:name"].apply(tuple).nunique()

    print(f"\nTop-{k} variant model  ({total_distinct} distinct variants in log)")
    for v in variants:
        print(f"\n  {v.variant_id}  ({v.case_count} cases)")
        for i, e in enumerate(v.sequence, 1):
            print(f"    {i:02d}. {e}")

    # ── 3. Variant overview CSV ───────────────────────────────────────────────
    build_variant_overview(variants).to_csv(out_dir / "variants_overview.csv", index=False)

    # ── 4. Build event objects ────────────────────────────────────────────────
    print("\nBuilding event objects …")
    events = build_events(df, variants)
    print(f"  Total events: {len(events)}")

    # ── 5. Synthetic signal ───────────────────────────────────────────────────
    print("\nGenerating synthetic signal …")
    sig_path, signal_df = generate_signal(events, out_dir)
    print(f"  Saved → {sig_path.name}")

    # ── 6. Cost table ─────────────────────────────────────────────────────────
    cost_table = build_cost_table(events)
    cost_table.to_csv(out_dir / "cost_table.csv", index=False)
    print("\nCost table (top 12 rows):")
    print(cost_table.head(12).to_string(index=False))

    # ── 7. Event occurrence matrix ────────────────────────────────────────────
    print("\nBuilding 30-min event matrix …")
    matrix = build_event_matrix(events, signal_df)
    matrix.to_csv(out_dir / "event_matrix_30min.csv", index=False)
    states = [c for c in matrix.columns if c != "timestamp"]
    print(f"  States in matrix: {states}")

    # ── 8. Markov chains ──────────────────────────────────────────────────────
    print("\nBuilding Markov chains …")

    # Pooled chain — all variants mixed; probabilities < 1 where paths diverge
    pooled = build_markov_chain(events)
    pooled.to_csv(out_dir / "markov_chain_pooled.csv", index=False)
    plot_markov_chain(pooled,
                      title=f"Pooled Markov chain ({k} variant{'s' if k > 1 else ''})",
                      output_path=out_dir / "markov_chain_pooled.png",
                      sequence_order=None)

    # Per-variant chains — within one variant all probabilities are ≈ 1
    for v in variants:
        v_events = [e for e in events if e.variant_id == v.variant_id]
        v_chain  = build_markov_chain(v_events)
        v_chain.to_csv(out_dir / f"{v.variant_id}_markov_chain.csv", index=False)
        plot_markov_chain(v_chain,
                          title=f"{v.variant_id} — transition probabilities",
                          output_path=out_dir / f"{v.variant_id}_markov_chain.png",
                          sequence_order=list(v.sequence))

    # ── 9. Energy learning ────────────────────────────────────────────────────
    print("\nRunning coordinate-descent energy learning …")
    summary = build_learning_summary(signal_df, matrix, events, states)
    summary.to_csv(out_dir / "learning_summary.csv", index=False)

    print("\n── Energy learning summary ──────────────────────────────────────────────")
    print(summary.to_string(index=False))
    print("─────────────────────────────────────────────────────────────────────────")
    print(f"\nAll outputs → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Event-based Markov energy model — BPI Challenge 2019."
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of top variants to model (default: 3).",
    )
    main(k=parser.parse_args().k)
