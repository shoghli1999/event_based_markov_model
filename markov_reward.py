"""
markov_reward.py
────────────────
Markov Reward Process (MRP) layer on top of MarkovModel_clean.py.

Idea
────
  • MarkovModel_clean.py already produces, per run:
        - transition probabilities   (markov_chain_pooled.csv, variant_*_markov_chain.csv)
        - per-state energy            (learning_summary.csv: true + recovered)
  • A Markov Reward Process attaches an energy (reward) to each state and lets the
    transition probabilities decide how often each state is visited.  Multiplying
    expected visits by per-state energy gives the EXPECTED ENERGY PER CASE — a number
    the plain least-squares baseline cannot produce.

Why this is the honest "Option A"
─────────────────────────────────
  • The five per-state energy VALUES are only identifiable from the binned signal,
    so they are read from learning_summary.csv (same route as the baseline).
  • The transition PROBABILITIES (estimated by counting) then turn those energies
    into per-case energy.  The pooled chain must reproduce the case-weighted average
    of the per-variant chains — that exact match is the validation.

This script reads ONLY the CSVs already written by MarkovModel_clean.py
(generated_signals_k{k}/), so it runs in milliseconds and never re-reads the XES log.

Usage
─────
  python markov_reward.py            # reads generated_signals_k3/  (default)
  python markov_reward.py --k 1      # reads generated_signals_k1/
  python markov_reward.py --energy true   # use ground-truth energies instead of recovered
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ─── Loading the clean-model outputs ───────────────────────────────────────────

def load_state_energy(out_dir: Path, energy_col: str) -> Dict[str, float]:
    """Per-state energy from learning_summary.csv (drops the AVERAGE row).

    energy_col: 'learned_noisy' (model output, default) or 'true_energy' (ground truth).
    """
    ls = pd.read_csv(out_dir / "learning_summary.csv")
    ls = ls[ls["state"] != "AVERAGE"]
    return dict(zip(ls["state"], ls[energy_col].astype(float)))


def load_variants(out_dir: Path) -> pd.DataFrame:
    """variant_id, case_count and the start state (first event of each variant)."""
    ov = pd.read_csv(out_dir / "variants_overview.csv")
    ov["start_state"] = ov["event_sequence"].str.split(" → ").str[0]
    return ov


# ─── Markov Reward Process core ─────────────────────────────────────────────────

def expected_visits(chain_csv: Path, init_dist: Dict[str, float]) -> Dict[str, float]:
    """Expected number of visits to each state per case, before absorption in END.

    Uses the fundamental matrix N = (I − Q)⁻¹ of the absorbing chain, where Q holds
    the transition probabilities between transient (non-END) states.  Visits are
    averaged over the supplied initial-state distribution (weighted by case counts
    for the pooled chain, or a single start state for one variant).
    """
    ch = pd.read_csv(chain_csv)
    states = [s for s in pd.unique(ch["state"]) if s != "END"]   # transient states
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)

    Q = np.zeros((n, n))
    for _, r in ch.iterrows():
        if r["state"] in idx and r["next_state"] in idx:        # transient → transient only
            Q[idx[r["state"]], idx[r["next_state"]]] = float(r["probability"])

    N = np.linalg.inv(np.eye(n) - Q)                            # expected visits matrix
    p0 = np.array([init_dist.get(s, 0.0) for s in states])     # initial-state distribution
    visits = p0 @ N
    return dict(zip(states, visits))


def expected_energy_per_case(visits: Dict[str, float], energy: Dict[str, float]) -> float:
    """Σ  expected_visits(state) × energy(state)  — the Markov reward."""
    return float(sum(visits[s] * energy.get(s, 0.0) for s in visits))


# ─── Assembly ───────────────────────────────────────────────────────────────────

def build_reward_summary(
    out_dir: Path, energy: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_case_df, pooled_visits_df).

    per_case_df rows: each variant, POOLED (chain-derived), WEIGHTED_AVG (sanity check).
    """
    variants = load_variants(out_dir)
    total_cases = int(variants["case_count"].sum())

    rows: List[dict] = []
    weighted = 0.0
    for _, v in variants.iterrows():
        vid, start, cases = v["variant_id"], v["start_state"], int(v["case_count"])
        visits = expected_visits(out_dir / f"{vid}_markov_chain.csv", {start: 1.0})
        e_case = expected_energy_per_case(visits, energy)
        weighted += e_case * cases
        rows.append({
            "chain":                  vid,
            "case_count":             cases,
            "start_state":            start,
            "expected_energy_per_case": round(e_case, 4),
        })

    # Pooled chain — initial distribution = each variant's start weighted by case count
    init = (
        variants.groupby("start_state")["case_count"].sum() / total_cases
    ).to_dict()
    pooled_visits = expected_visits(out_dir / "markov_chain_pooled.csv", init)
    e_pooled = expected_energy_per_case(pooled_visits, energy)

    rows.append({
        "chain": "POOLED (from probabilities)", "case_count": total_cases,
        "start_state": ", ".join(init), "expected_energy_per_case": round(e_pooled, 4),
    })
    rows.append({
        "chain": "WEIGHTED_AVG (variant check)", "case_count": total_cases,
        "start_state": "—", "expected_energy_per_case": round(weighted / total_cases, 4),
    })

    per_case_df = pd.DataFrame(rows)
    pooled_visits_df = (
        pd.DataFrame({"state": list(pooled_visits),
                      "expected_visits_per_case": [round(pooled_visits[s], 4) for s in pooled_visits]})
        .sort_values("state").reset_index(drop=True)
    )
    return per_case_df, pooled_visits_df


def build_recovery_table(out_dir: Path) -> pd.DataFrame:
    """Option-A comparison: true vs recovered per-state energy (same target as baseline)."""
    ls = pd.read_csv(out_dir / "learning_summary.csv")
    ls = ls[ls["state"] != "AVERAGE"].copy()
    ls["abs_error"] = (ls["learned_noisy"] - ls["true_energy"]).abs().round(6)
    return ls[["state", "true_energy", "learned_noisy", "abs_error"]].rename(
        columns={"learned_noisy": "markov_recovered"}
    )


# ─── Plotting (same networkx style as MarkovModel_clean.py) ─────────────────────

def _node_label(name: str, energy: Dict[str, float], visits: Dict[str, float],
                max_len: int = 20) -> str:
    """State name + its energy (E) and expected visits per case (n)."""
    if name == "END":
        return "END"
    short = name if len(name) <= max_len else name[: max_len - 1] + "…"
    if name in energy:
        return f"{short}\nE={energy[name]:.1f}  n={visits.get(name, 0.0):.2f}"
    return short


def _linear_pos(nodes, sequence) -> Dict[str, Tuple[float, float]]:
    """Left-to-right layout following the variant's event order (END last)."""
    rank = {s: i for i, s in enumerate(sequence)}
    rank["END"] = len(sequence)
    top = max(rank.values())
    pos, extra = {}, 0.0
    for n in nodes:
        if n in rank:
            pos[n] = (float(rank[n]), 0.0)
        else:
            pos[n] = (float(top + 1), extra)
            extra += 0.4
    return pos


def plot_reward_chain(chain_csv: Path, energy: Dict[str, float], visits: Dict[str, float],
                      title: str, output_path: Path, sequence_order=None,
                      min_prob: float = 0.001) -> None:
    """Markov reward chain: nodes carry energy + visits, edges carry probability."""
    ch = pd.read_csv(chain_csv)
    edges = ch[ch["probability"] >= min_prob]
    if edges.empty:
        return

    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["state"]), str(r["next_state"]), label=f"{float(r['probability']):.3f}")

    pos = (_linear_pos(list(G.nodes), sequence_order)
           if sequence_order is not None else nx.spring_layout(G, seed=42, k=2.0))

    fig, ax = plt.subplots(figsize=(14, 7))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3600,
                           node_color="#d9ecff", edgecolors="#1f4e79", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                            labels={n: _node_label(n, energy, visits) for n in G.nodes})
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18,
                           width=1.8, connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "label"), font_size=8, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Plot saved → {output_path.name}")


def plot_energy_per_case(per_case_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of expected energy per case (pooled bar highlighted)."""
    df = per_case_df[~per_case_df["chain"].str.startswith("WEIGHTED")].copy()
    labels = df["chain"].str.replace(" (from probabilities)", "", regex=False)
    vals = df["expected_energy_per_case"]
    colors = ["#4c78a8"] * (len(df) - 1) + ["#e45756"]   # pooled highlighted in red

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="#1f4e79", linewidth=1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Expected energy per case")
    ax.set_title("Markov reward process — expected energy per case")
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Plot saved → {output_path.name}")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main(k: int, energy_col: str) -> None:
    # reads CSVs from MarkovModel_clean.py — does not reload the XES log
    thesis_dir = Path(__file__).resolve().parent
    out_dir = thesis_dir / f"generated_signals_k{k}"
    if not out_dir.exists():
        raise SystemExit(
            f"{out_dir} not found — run  python MarkovModel_clean.py --k {k}  first."
        )

    # 1. load per-state energies (recovered from signal or ground truth)
    energy = load_state_energy(out_dir, energy_col)

    print(f"Markov Reward Process  (k={k}, energy source: {energy_col})")
    print("=" * 74)

    # 2. show how well the signal-based recovery matches true energies
    print("\n── Option A: per-state energy recovery (values come from the signal) ──")
    recovery = build_recovery_table(out_dir)
    print(recovery.to_string(index=False))

    # 3. compute expected visits and expected energy per case via the Markov reward process
    per_case_df, pooled_visits_df = build_reward_summary(out_dir, energy)

    print("\n── Expected visits per case  (pooled chain — uses the probabilities) ──")
    print(pooled_visits_df.to_string(index=False))

    print("\n── Expected ENERGY PER CASE  (Markov reward = visits × energy) ──")
    print(per_case_df.to_string(index=False))

    # 4. sanity check: pooled chain must match the case-weighted variant average
    pooled = per_case_df.loc[per_case_df["chain"].str.startswith("POOLED"), "expected_energy_per_case"].iloc[0]
    wavg   = per_case_df.loc[per_case_df["chain"].str.startswith("WEIGHTED"), "expected_energy_per_case"].iloc[0]
    gap = abs(pooled - wavg)
    print(f"\nValidation: pooled {pooled:.4f}  vs  weighted-average {wavg:.4f}  →  gap {gap:.2e}", end="  ")
    print("✓ probabilities reproduce the mixture" if gap < 1e-3 else "✗ mismatch")

    # 5. save reward-process tables
    recovery.to_csv(out_dir / "markov_reward_recovery.csv", index=False)
    per_case_df.to_csv(out_dir / "markov_reward_per_case.csv", index=False)
    pooled_visits_df.to_csv(out_dir / "markov_reward_visits.csv", index=False)
    print(f"\nSaved → {out_dir}/markov_reward_per_case.csv, _visits.csv, _recovery.csv")

    # 6. plot reward chains (energy + visits on nodes, probabilities on edges)
    print("\nPlotting …")
    variants = load_variants(out_dir)
    for _, v in variants.iterrows():
        vid = v["variant_id"]
        chain_csv = out_dir / f"{vid}_markov_chain.csv"
        visits = expected_visits(chain_csv, {v["start_state"]: 1.0})
        plot_reward_chain(
            chain_csv, energy, visits,
            title=f"{vid} — reward chain  (E = energy, n = visits/case, edge = probability)",
            output_path=out_dir / f"{vid}_reward_chain.png",
            sequence_order=v["event_sequence"].split(" → "),
        )

    total = int(variants["case_count"].sum())
    init = (variants.groupby("start_state")["case_count"].sum() / total).to_dict()
    pooled_visits = expected_visits(out_dir / "markov_chain_pooled.csv", init)
    plot_reward_chain(
        out_dir / "markov_chain_pooled.csv", energy, pooled_visits,
        title="Pooled reward chain  (E = energy, n = expected visits/case, edge = probability)",
        output_path=out_dir / "markov_reward_chain_pooled.png",
        sequence_order=None,
    )
    plot_energy_per_case(per_case_df, out_dir / "markov_reward_per_case.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markov Reward Process on MarkovModel_clean.py outputs.")
    parser.add_argument("--k", type=int, default=3, help="Which generated_signals_k{k}/ folder to read (default 3).")
    parser.add_argument("--energy", choices=["learned", "true"], default="learned",
                        help="Energy source: 'learned' (recovered, default) or 'true' (ground truth).")
    args = parser.parse_args()
    main(k=args.k, energy_col="learned_noisy" if args.energy == "learned" else "true_energy")
