"""
compare_optimizers.py
─────────────────────
Compares three ways of solving the SAME energy-learning problem:

    find w  such that   X w  ≈  y

    X = event_matrix_30min.csv   (rows = 30-min intervals, cols = event types)
    y = signal.csv "clean"/"signal" minus the baseline
    w = energy per event type  (what we want to recover)

Methods compared
────────────────
  1. Coordinate descent  — the "guess and refine" loop used in MarkovModel_clean.py
  2. Gradient descent    — the standard alternative
  3. Exact least squares — np.linalg.lstsq, the true optimum (reference answer)


Usage
─────
  python compare_optimizers.py              # uses generated_signals_k3/
  python compare_optimizers.py --k 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_problem(out_dir: Path, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build X (event counts per interval) and y (energy per interval).

    target: 'clean' (no noise) or 'signal' (with noise).
    The sine baseline is subtracted so only the event-cost part remains.
    """
    matrix = pd.read_csv(out_dir / "event_matrix_30min.csv")
    signal = pd.read_csv(out_dir / "signal.csv")

    states = [c for c in matrix.columns if c != "timestamp"]
    X = matrix[states].to_numpy(dtype=float)
    y = (signal[target] - signal["baseline"]).to_numpy(dtype=float)
    return X, y, states


def load_true_energy(out_dir: Path, states: List[str]) -> np.ndarray:
    """Ground-truth energy per state, as written by MarkovModel_clean.py."""
    ls = pd.read_csv(out_dir / "learning_summary.csv")
    ls = ls[ls["state"] != "AVERAGE"].set_index("state")["true_energy"]
    return np.array([float(ls[s]) for s in states])


# ─── Method 1: coordinate descent (same maths as MarkovModel_clean.py) ────────

def coordinate_descent(X: np.ndarray, y: np.ndarray, iterations: int) -> np.ndarray:
    """Update one weight at a time, each to its exact best value.

        w_j  ←  col_j · (y − contribution of all other columns) / (col_j · col_j)

    No step size to choose: each update is the exact minimiser along that axis.
    """
    n_features = X.shape[1]
    w = np.ones(n_features)
    denom = (X ** 2).sum(axis=0)
    residual = y - X @ w                       # kept up to date incrementally
    for _ in range(iterations):
        for j in range(n_features):
            if denom[j] == 0.0:
                continue
            residual += X[:, j] * w[j]         # remove column j's contribution
            w[j] = float(X[:, j] @ residual / denom[j])
            residual -= X[:, j] * w[j]         # put the new one back
    return w


# ─── Method 2: gradient descent ───────────────────────────────────────────────

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    iterations: int,
    lr: float | None = None,
) -> np.ndarray:
    """Move all weights together, downhill, by a fixed step size.

        w  ←  w − lr · Xᵀ(X w − y)

    If lr is None, use the largest safe step 1/L, where L is the biggest
    eigenvalue of XᵀX.  Anything bigger than 2/L diverges.
    """
    if lr is None:
        L = float(np.linalg.eigvalsh(X.T @ X).max())
        lr = 1.0 / L
    w = np.ones(X.shape[1])
    for _ in range(iterations):
        w -= lr * (X.T @ (X @ w - y))
    return w


def gradient_descent_scaled(
    X: np.ndarray,
    y: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Gradient descent after rescaling each column to unit norm.

    Rescaling makes the columns comparable in size, which is the standard fix
    for slow gradient descent.  The weights are converted back at the end.
    """
    scale = np.sqrt((X ** 2).sum(axis=0))
    scale[scale == 0.0] = 1.0
    Xs = X / scale
    w_scaled = gradient_descent(Xs, y, iterations)
    return w_scaled / scale


# ─── Method 3: exact solution ─────────────────────────────────────────────────

def exact_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The true optimum, solved directly. This is what both iterative methods
    are trying to reach."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ─── Measuring ────────────────────────────────────────────────────────────────

def max_error(w: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(w - truth).max())


def timed(fn, *args, **kwargs) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    w = fn(*args, **kwargs)
    return w, time.perf_counter() - t0


def iterations_to_reach(fn, X, y, truth, tol: float, budget: List[int]) -> str:
    """Smallest iteration count in `budget` that gets max error below tol."""
    for it in budget:
        if max_error(fn(X, y, it), truth) < tol:
            return str(it)
    return f">{budget[-1]}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(k: int) -> None:
    out_dir = Path(__file__).resolve().parent / f"generated_signals_k{k}"
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} not found — run MarkovModel_clean.py --k {k} first.")

    for target, label in [("clean", "CLEAN signal (no noise)"),
                          ("signal", "NOISY signal")]:
        X, y, states = load_problem(out_dir, target)
        truth = load_true_energy(out_dir, states)

        XtX = X.T @ X
        eig = np.linalg.eigvalsh(XtX)
        condition = float(eig.max() / eig.min())

        print("=" * 78)
        print(f"{label}   —   {X.shape[0]} intervals × {X.shape[1]} event types")
        print("=" * 78)
        print(f"  condition number of XᵀX : {condition:,.1f}")
        print(f"  (how stretched the problem is; big = gradient descent struggles)")
        print()

        rows = []
        for name, w, secs in [
            ("Coordinate descent (20 passes)",  *timed(coordinate_descent, X, y, 20)),
            ("Gradient descent (20 steps)",     *timed(gradient_descent, X, y, 20)),
            ("Gradient descent (1,000 steps)",  *timed(gradient_descent, X, y, 1000)),
            ("Gradient descent (100,000)",      *timed(gradient_descent, X, y, 100_000)),
            ("Gradient descent, scaled (1,000)", *timed(gradient_descent_scaled, X, y, 1000)),
            ("Exact least squares",             *timed(exact_least_squares, X, y)),
        ]:
            rows.append({
                "method":        name,
                "max_error":     f"{max_error(w, truth):.3e}",
                "seconds":       f"{secs:.4f}",
            })
        print(pd.DataFrame(rows).to_string(index=False))

        budget = [1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000, 20000, 100000]
        print()
        print("  passes/steps needed to get every energy within 0.001 of truth:")
        print(f"    coordinate descent : "
              f"{iterations_to_reach(coordinate_descent, X, y, truth, 1e-3, budget)}")
        print(f"    gradient descent   : "
              f"{iterations_to_reach(gradient_descent, X, y, truth, 1e-3, budget)}")
        print(f"    gradient descent (scaled) : "
              f"{iterations_to_reach(gradient_descent_scaled, X, y, truth, 1e-3, budget)}")
        print()

    print("Reminder: coordinate descent and gradient descent are two routes to the")
    print("SAME answer (the exact least-squares solution). They differ only in how")
    print("fast they get there, not in where they end up.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare coordinate vs gradient descent.")
    p.add_argument("--k", type=int, default=3, help="which generated_signals_k{k} folder")
    main(p.parse_args().k)
