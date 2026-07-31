"""
regression_comparison.py
────────────────────────
Compares the linear-regression baseline against the coordinate-descent method
used in MarkovModel_clean.py, plus ridge regression, under a proper time-based
train/test split.

The linear-regression baseline
──────────────────────────────
This reproduces my supervisor's implementation in "Event Log Manager/
event_cost_eval.py" (estimate_energy_per_event):

    X = column_stack([ones, event_matrix.T])
    model = LinearRegression(fit_intercept=False, positive=True)

Two things matter there:
  • positive=True  → energies are constrained to be >= 0 (non-negative least
    squares). Energy cannot be negative, so this is a real modelling choice,
    not a detail.
  • the ones-column   → the background load is ESTIMATED as an intercept.

That second point is the important difference from MarkovModel_clean.py, which
subtracts the *true* generated baseline before fitting. In practice the true
baseline is unknown, so subtracting it makes the task easier than it really is.

This script therefore evaluates every method under BOTH setups:

    "oracle"  y = signal − true baseline, no intercept   (current thesis code)
    "honest"  y = signal, intercept estimated            (baseline setup)

Reads only the CSVs written by MarkovModel_clean.py — the 695 MB XES log is
never touched.

Usage
─────
  python regression_comparison.py            # generated_signals_k3/
  python regression_comparison.py --k 2
  python regression_comparison.py --train-fraction 0.8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Ridge penalties to sweep. 0 would be plain OLS, so we start just above it.
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Coordinate-descent passes, matching LEARN_ITERS in MarkovModel_clean.py.
CD_ITERS = 20


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data(out_dir: Path) -> Tuple[np.ndarray, pd.DataFrame, List[str], np.ndarray]:
    """Return X (interval × event-type counts), the signal frame, state names
    and the ground-truth energy per state."""
    matrix = pd.read_csv(out_dir / "event_matrix_30min.csv")
    signal = pd.read_csv(out_dir / "signal.csv")
    states = [c for c in matrix.columns if c != "timestamp"]

    if len(matrix) != len(signal):
        raise SystemExit("event_matrix and signal have different lengths — rerun MarkovModel_clean.py")

    ls = pd.read_csv(out_dir / "learning_summary.csv")
    ls = ls[ls["state"] != "AVERAGE"].set_index("state")["true_energy"]
    truth = np.array([float(ls[s]) for s in states])

    return matrix[states].to_numpy(dtype=float), signal, states, truth


def time_split(n: int, train_fraction: float) -> Tuple[slice, slice]:
    """Chronological split: the first part trains, the later part tests.

    Rows are already in time order, so slicing is the split. A random split
    would leak future information into training, which is why we do not use one.
    """
    cut = int(round(n * train_fraction))
    return slice(0, cut), slice(cut, n)


# ─── Estimators ───────────────────────────────────────────────────────────────

def _design(X: np.ndarray, intercept: bool) -> np.ndarray:
    """Attach a ones-column when the background load must be estimated."""
    return np.column_stack([np.ones(len(X)), X]) if intercept else X


def fit_linreg_baseline(X: np.ndarray, y: np.ndarray, intercept: bool) -> np.ndarray:
    """Non-negative least squares, exactly as in event_cost_eval.py
    (my supervisor's implementation)."""
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(_design(X, intercept), y)
    return model.coef_


def fit_ols(X: np.ndarray, y: np.ndarray, intercept: bool) -> np.ndarray:
    """Unconstrained least squares, solved directly."""
    return np.linalg.lstsq(_design(X, intercept), y, rcond=None)[0]


def fit_ridge(X: np.ndarray, y: np.ndarray, intercept: bool, alpha: float) -> np.ndarray:
    """Least squares with an L2 penalty that shrinks the energies towards zero.

    Solved as  w = (DᵀD + alpha·P)⁻¹ Dᵀy  with P = diag(0, 1, 1, …).

    The zero in P is what keeps the intercept out of the penalty. sklearn's
    Ridge cannot express this directly: with fit_intercept=False it penalises
    every column of the matrix it is given, including a manually attached
    ones-column, which would shrink the background load towards zero along with
    the event energies.
    """
    D = _design(X, intercept)
    P = np.eye(D.shape[1])
    if intercept:
        P[0, 0] = 0.0          # background load is not shrunk
    return np.linalg.solve(D.T @ D + alpha * P, D.T @ y)


def fit_coordinate_descent(X: np.ndarray, y: np.ndarray, intercept: bool) -> np.ndarray:
    """The 'guess and refine' loop from MarkovModel_clean.py.

    One weight at a time, each set to its exact best value given the others.
    Residual is updated incrementally instead of rebuilt, which is the same
    maths but much faster than the version in MarkovModel_clean.py.
    """
    D = _design(X, intercept)
    w = np.ones(D.shape[1])
    denom = (D ** 2).sum(axis=0)
    residual = y - D @ w
    for _ in range(CD_ITERS):
        for j in range(D.shape[1]):
            if denom[j] == 0.0:
                continue
            residual += D[:, j] * w[j]
            w[j] = float(D[:, j] @ residual / denom[j])
            residual -= D[:, j] * w[j]
    return w


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score(
    coef: np.ndarray,
    intercept: bool,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    truth: np.ndarray,
) -> Dict[str, float]:
    """Energy-recovery error plus held-out predictive accuracy."""
    energies = coef[1:] if intercept else coef
    pred_te = _design(X_te, intercept) @ coef
    pred_tr = _design(X_tr, intercept) @ coef
    return {
        "intercept":     float(coef[0]) if intercept else np.nan,
        "max_abs_err":   float(np.abs(energies - truth).max()),
        "mean_abs_err":  float(np.abs(energies - truth).mean()),
        "train_r2":      float(r2_score(y_tr, pred_tr)),
        "test_r2":       float(r2_score(y_te, pred_te)),
        "test_rmse":     float(np.sqrt(mean_squared_error(y_te, pred_te))),
    }


def run_setup(
    name: str,
    X: np.ndarray, y: np.ndarray,
    intercept: bool,
    truth: np.ndarray,
    states: List[str],
    train_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every method on the training slice and score it on the test slice."""
    tr, te = time_split(len(X), train_fraction)
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

    fits = [
        ("Linear regression baseline (NNLS)", fit_linreg_baseline(X_tr, y_tr, intercept)),
        ("Direct least squares (OLS)",       fit_ols(X_tr, y_tr, intercept)),
        ("Coordinate descent (20 passes)",   fit_coordinate_descent(X_tr, y_tr, intercept)),
    ]
    fits += [(f"Ridge (alpha={a:g})", fit_ridge(X_tr, y_tr, intercept, a)) for a in RIDGE_ALPHAS]

    rows = []
    for label, coef in fits:
        rows.append({"setup": name, "method": label,
                     **score(coef, intercept, X_tr, y_tr, X_te, y_te, truth)})

    # Per-state energies for the three headline methods
    per_state = pd.DataFrame({"state": states, "true_energy": truth})
    for label, coef in fits[:3]:
        per_state[label] = (coef[1:] if intercept else coef)

    return pd.DataFrame(rows), per_state


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(k: int, train_fraction: float) -> None:
    out_dir = Path(__file__).resolve().parent / f"generated_signals_k{k}"
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} not found — run MarkovModel_clean.py --k {k} first.")

    X, signal, states, truth = load_data(out_dir)
    n_tr = int(round(len(X) * train_fraction))

    print(f"{len(X)} intervals × {len(states)} event types")
    print(f"time split: first {n_tr} intervals train, last {len(X) - n_tr} test")
    print(f"(train ends {signal['timestamp'].iloc[n_tr - 1]})\n")

    setups = [
        # Current thesis code: the true sine baseline is removed beforehand.
        ("ORACLE BASELINE  (true sine subtracted, no intercept)",
         (signal["signal"] - signal["baseline"]).to_numpy(float), False),
        # Linear-regression baseline setup. Note this is NOT a realistic baseline model: a single
        # constant cannot follow a daily sine, so the unmodelled wave stays in the
        # residual. Calling it "honest" would overstate it — it is one constant.
        ("CONSTANT BASELINE  (raw signal, single intercept)",
         signal["signal"].to_numpy(float), True),
    ]

    all_metrics, all_states = [], []
    for name, y, intercept in setups:
        metrics, per_state = run_setup(name, X, y, intercept, truth, states, train_fraction)
        all_metrics.append(metrics)
        all_states.append(per_state.assign(setup=name))

        print("=" * 100)
        print(name)
        print("=" * 100)
        show = metrics.drop(columns=["setup"])
        if not intercept:
            show = show.drop(columns=["intercept"])
        print(show.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print("\n  recovered energy per state:")
        print(per_state.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print()

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(out_dir / "regression_comparison_metrics.csv", index=False)
    pd.concat(all_states, ignore_index=True).to_csv(
        out_dir / "regression_comparison_energies.csv", index=False)
    print(f"saved → {out_dir.name}/regression_comparison_metrics.csv")
    print(f"saved → {out_dir.name}/regression_comparison_energies.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Linear regression baseline vs coordinate descent vs ridge.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--train-fraction", type=float, default=0.7)
    main(p.parse_args().k, p.parse_args().train_fraction)
