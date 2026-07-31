"""
hmm_variants.py
───────────────
Hidden Markov Model over one or more process variants (Baum-Welch + Viterbi,
written from scratch). Generalises hmm_variant1.py.

Why more than one variant matters
─────────────────────────────────
With variant 1 alone the activity order is fixed, so step number alone already
identifies the activity and an HMM looks unnecessary. Variants 1 and 2 use the
SAME five activities but swap steps 2 and 3:

    variant 1:  Create PO → Vendor creates invoice → Record Goods Receipt → ...
    variant 2:  Create PO → Record Goods Receipt → Vendor creates invoice → ...

So step number no longer determines the activity, and the hidden states are the
ACTIVITIES themselves rather than the positions. The transition matrix now
genuinely branches, which is what an HMM is for.

Four methods on identical data
──────────────────────────────
  1. Linear regression baseline — labels KNOWN. One-hot design matrix, so the least
     squares coefficients are the per-activity means. Best-case reference. Same
     family as my supervisor's implementation in event_cost_eval.py.
  2. Position baseline       — labels known in TRAINING only; at test time it
     predicts the most common activity for that step number. This is the
     strongest form of the objection "you could just use the step number".
  3. HMM                     — labels HIDDEN, order modelled by transitions.
  4. Gaussian mixture        — labels HIDDEN, order IGNORED. Isolates what the
     Markov part contributes.

Cases are drawn in proportion to the real case counts in variants_overview.csv,
so the branching probabilities match the log. The 695 MB XES is never read.

Usage
─────
  python hmm_variants.py                      # variants 1+2+3, noise sweep
  python hmm_variants.py --variants 1         # reproduces hmm_variant1.py
  python hmm_variants.py --variants 1,2 --noise 5   # equal-length subset
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

# Same generator constants as MarkovModel_clean.py
BASE_COSTS: Dict[str, float] = {
    "Record Goods Receipt": 99, "Create Purchase Order Item": 98,
    "Record Invoice Receipt": 97, "Vendor creates invoice": 54, "Clear Invoice": 56,
}
DURATION_MEAN, DURATION_STD = 180.0, 20.0
DURATION_MIN, DURATION_MAX = 120.0, 240.0
DURATION_SCALE = 0.01

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.1      # used only to choose the number of hidden states
K_CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
SELECT_TOL = 0.01       # nats per observation; smallest K within this of the best wins
EM_ITERS = 200
EM_TOL = 1e-8
N_RESTARTS = 8          # restarts for the final model
SEARCH_RESTARTS = 3     # restarts per candidate during the K search
DATA_SEED = 42
NOISE_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_variants(out_dir: Path, wanted: List[int]) -> Tuple[List[List[str]], np.ndarray, List[str]]:
    """Activity sequences and case-count weights for the requested variants."""
    ov = pd.read_csv(out_dir / "variants_overview.csv").set_index("variant_id")

    sequences, counts = [], []
    for v in wanted:
        row = ov.loc[f"variant_{v}"]
        sequences.append([a.strip() for a in row["event_sequence"].split("→")])
        counts.append(float(row["case_count"]))

    # Variants may differ in length — variant 3 has 2 steps against 5 for
    # variants 1 and 2. Traces are right-padded to the longest one and carry a
    # mask, so no variant has to be dropped.

    # Stable activity ordering: first appearance across the chosen variants
    activities: List[str] = []
    for seq in sequences:
        for a in seq:
            if a not in activities:
                activities.append(a)

    weights = np.array(counts) / sum(counts)
    return sequences, weights, activities


def simulate_cases(sequences: List[List[str]], weights: np.ndarray, activities: List[str],
                   n_cases: int, noise_std: float,
                   seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One energy reading per step, per case.

    Shorter variants are right-padded to the longest trace length. `mask` says
    which entries are real; padded entries carry filler values that every
    downstream step ignores.

    Returns
      obs    (n_cases, T) observed energy — all the hidden methods may see
      clean  (n_cases, T) energy before measurement noise
      labels (n_cases, T) true ACTIVITY index — used only for scoring
      mask   (n_cases, T) True where the case really has a step
    """
    rng = np.random.default_rng(seed)
    T = max(len(s) for s in sequences)
    a_idx = {a: i for i, a in enumerate(activities)}

    # Which variant each case follows, in proportion to the real case counts
    choice = rng.choice(len(sequences), size=n_cases, p=weights)
    seq_idx = np.zeros((len(sequences), T), dtype=int)
    seq_mask = np.zeros((len(sequences), T), dtype=bool)
    for i, seq in enumerate(sequences):
        seq_idx[i, :len(seq)] = [a_idx[a] for a in seq]
        seq_mask[i, :len(seq)] = True

    labels = seq_idx[choice]
    mask = seq_mask[choice]

    base = np.array([BASE_COSTS[a] for a in activities])
    duration = np.clip(rng.normal(DURATION_MEAN, DURATION_STD, (n_cases, T)),
                       DURATION_MIN, DURATION_MAX)
    clean = base[labels] + duration * DURATION_SCALE
    obs = clean + rng.normal(0.0, noise_std, (n_cases, T))

    # Padded slots must not look like data to anything downstream.
    obs = np.where(mask, obs, 0.0)
    clean = np.where(mask, clean, np.nan)
    return obs, clean, labels, mask


# ─── HMM: Baum-Welch ──────────────────────────────────────────────────────────

def _gaussian_pdf(obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Emission likelihood B[n, t, k]."""
    sigma = np.maximum(sigma, 1e-6)
    z = (obs[:, :, None] - mu[None, None, :]) / sigma[None, None, :]
    return np.exp(-0.5 * z ** 2) / (sigma[None, None, :] * np.sqrt(2 * np.pi))


def _forward_backward(B: np.ndarray, pi: np.ndarray, A: np.ndarray,
                      mask: np.ndarray | None = None):
    """Scaled forward-backward; scaling prevents underflow.

    With `mask`, traces shorter than T are handled without cutting the batch up.
    Once a case has ended, alpha is frozen at its last real step and its scaling
    factor is set to 1, so padded steps add log 1 = 0 to the likelihood and the
    chain is not allowed to keep walking through the transition matrix. Beta is
    carried backwards unchanged over the same region, because there are no
    future observations there to condition on.
    """
    N, T, K = B.shape
    if mask is None:
        mask = np.ones((N, T), dtype=bool)

    alpha = np.zeros((N, T, K))
    beta = np.zeros((N, T, K))
    c = np.ones((N, T))

    alpha[:, 0] = pi[None, :] * B[:, 0]
    c[:, 0] = np.maximum(alpha[:, 0].sum(axis=1), 1e-300)
    alpha[:, 0] /= c[:, 0, None]
    for t in range(1, T):
        step = (alpha[:, t - 1] @ A) * B[:, t]
        s = np.maximum(step.sum(axis=1), 1e-300)
        live = mask[:, t]
        alpha[:, t] = np.where(live[:, None], step / s[:, None], alpha[:, t - 1])
        c[:, t] = np.where(live, s, 1.0)

    beta[:, T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        step = ((B[:, t + 1] * beta[:, t + 1]) @ A.T) / c[:, t + 1, None]
        beta[:, t] = np.where(mask[:, t + 1][:, None], step, beta[:, t + 1])

    return alpha, beta, c, float(np.log(c).sum())


def _init_params(obs: np.ndarray, K: int, rng, restart: int):
    """Starting point for one EM run.

    EM only finds a local optimum, and on this data a bad start has a specific
    failure mode: it learns a deterministic chain that uses the states as step
    numbers, fitting one wide Gaussian over two activities that share a step.
    That solution is stable but wrong, so restart 0 starts from a 1-D Gaussian
    mixture of the pooled readings, which already places the means correctly and
    leaves EM only the transition structure to discover.

    Later restarts stay random for comparison, but their transition matrices
    start near-uniform (large Dirichlet concentration) rather than spiky, so
    they are not pushed towards a deterministic chain from the first iteration.
    """
    flat = obs.reshape(-1, 1)
    if restart == 0:
        gmm = GaussianMixture(n_components=K, n_init=5, random_state=int(rng.integers(1 << 31)))
        gmm.fit(flat)
        mu = gmm.means_.ravel()
        sigma = np.sqrt(gmm.covariances_.ravel())
        pi = np.full(K, 1.0 / K)
        A = np.full((K, K), 1.0 / K)
    else:
        mu = rng.permutation(np.linspace(flat.min(), flat.max(), K))
        sigma = np.full(K, flat.std())
        pi = rng.dirichlet(np.full(K, 5.0))
        A = rng.dirichlet(np.full(K, 5.0), size=K)
    return mu, sigma, pi, A


def fit_hmm(obs: np.ndarray, K: int, seed: int, restarts: int | None = None,
            mask: np.ndarray | None = None):
    """Baum-Welch (EM), best of `restarts` starts, judged by log-likelihood.

    The K search refits the model once per candidate, so it passes a smaller
    restart count; the final model for the chosen K uses the full N_RESTARTS.

    `mask` marks the real steps when traces have different lengths. Padded slots
    are removed from every sufficient statistic, so they cannot pull the means,
    the variances or the transition counts.
    """
    rng = np.random.default_rng(seed)
    if mask is None:
        mask = np.ones(obs.shape, dtype=bool)
    real = obs[mask]                      # only genuine readings seed the start
    best = None

    for restart in range(restarts if restarts is not None else N_RESTARTS):
        mu, sigma, pi, A = _init_params(real, K, rng, restart)

        prev = -np.inf
        for _ in range(EM_ITERS):
            B = _gaussian_pdf(obs, mu, sigma)
            B = np.where(mask[:, :, None], B, 1.0)   # padded steps stay neutral
            alpha, beta, c, loglik = _forward_backward(B, pi, A, mask)

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=2, keepdims=True), 1e-300)
            gamma *= mask[:, :, None]

            # A transition only counts when both of its endpoints are real.
            pair = (mask[:, :-1] & mask[:, 1:])[:, :, None, None]
            xi = (alpha[:, :-1, :, None]
                  * A[None, None, :, :]
                  * (B[:, 1:, None, :] * beta[:, 1:, None, :])
                  / np.maximum(c[:, 1:, None, None], 1e-300)) * pair

            pi = gamma[:, 0, :].mean(axis=0)
            denom = (gamma[:, :-1, :] * (mask[:, 1:, None])).sum(axis=(0, 1))
            A = xi.sum(axis=(0, 1)) / np.maximum(denom[:, None], 1e-300)
            A /= np.maximum(A.sum(axis=1, keepdims=True), 1e-300)

            w = gamma.sum(axis=(0, 1))
            mu = (gamma * obs[:, :, None]).sum(axis=(0, 1)) / np.maximum(w, 1e-300)
            var = (gamma * (obs[:, :, None] - mu[None, None, :]) ** 2).sum(axis=(0, 1)) / np.maximum(w, 1e-300)
            sigma = np.sqrt(np.maximum(var, 1e-12))

            if abs(loglik - prev) < EM_TOL * max(1.0, abs(prev)):
                break
            prev = loglik

        if best is None or loglik > best[-1]:
            best = (pi, A, mu, sigma, loglik)
    return best


def viterbi(obs, pi, A, mu, sigma, mask: np.ndarray | None = None) -> np.ndarray:
    """Most likely hidden state sequence per case, in log space.

    Past the end of a short trace the recursion stops advancing: the score is
    held and the backpointer is the identity, so the traceback simply carries
    the last real state backwards and padded slots never steer the decoding.
    """
    N, T = obs.shape
    K = len(mu)
    if mask is None:
        mask = np.ones((N, T), dtype=bool)

    logB = np.log(np.maximum(_gaussian_pdf(obs, mu, sigma), 1e-300))
    logB = np.where(mask[:, :, None], logB, 0.0)
    logA = np.log(np.maximum(A, 1e-300))
    identity = np.arange(K)[None, :]

    delta = np.log(np.maximum(pi, 1e-300))[None, :] + logB[:, 0]
    psi = np.zeros((N, T, K), dtype=int)
    for t in range(1, T):
        scores = delta[:, :, None] + logA[None, :, :]
        live = mask[:, t][:, None]
        psi[:, t] = np.where(live, scores.argmax(axis=1), identity)
        delta = np.where(live, scores.max(axis=1) + logB[:, t], delta)

    path = np.zeros((N, T), dtype=int)
    path[:, T - 1] = delta.argmax(axis=1)
    for t in range(T - 2, -1, -1):
        path[:, t] = psi[np.arange(N), t + 1, path[:, t + 1]]
    return path


# ─── Aligning learned states with true activities ─────────────────────────────

def map_states_from_train(pred_tr: np.ndarray, labels_tr: np.ndarray,
                          K: int, n_act: int) -> np.ndarray:
    """Decide which activity each learned state stands for, using TRAINING ONLY.

    EM numbers its states arbitrarily, so some mapping is unavoidable before the
    predictions can be scored. Building it from the training decoding keeps the
    test set out of the evaluation entirely; an earlier version matched learned
    means against activity means computed over the whole data set, which leaked
    test information into the mapping.

    Each learned state is sent to the activity it covers most often in training.
    The mapping is many-to-one on purpose, so it still works when K differs from
    the number of activities — which it must, once K is chosen by validation.
    """
    mapping = np.zeros(K, dtype=int)
    for k in range(K):
        m = pred_tr == k
        mapping[k] = np.bincount(labels_tr[m], minlength=n_act).argmax() if m.any() else 0
    return mapping


def energy_per_activity(mu: np.ndarray, mapping: np.ndarray, pred_tr: np.ndarray,
                        n_act: int) -> np.ndarray:
    """Per-activity energy implied by the learned state means.

    When several states map to one activity their means are averaged, weighted
    by how often each state was decoded in training.
    """
    out = np.full(n_act, np.nan)
    for a in range(n_act):
        ks = np.flatnonzero(mapping == a)
        if len(ks) == 0:
            continue
        w = np.array([np.sum(pred_tr == k) for k in ks], dtype=float)
        out[a] = float(np.average(mu[ks], weights=w)) if w.sum() > 0 else float(mu[ks].mean())
    return out


def loglikelihood(obs: np.ndarray, pi, A, mu, sigma, mask=None) -> float:
    """Log-likelihood of held-out sequences under a fitted model."""
    B = _gaussian_pdf(obs, mu, sigma)
    if mask is not None:
        B = np.where(mask[:, :, None], B, 1.0)
    return _forward_backward(B, pi, A, mask)[3]


def select_k(obs_tr: np.ndarray, obs_val: np.ndarray, candidates: List[int],
             seed: int, mask_tr=None, mask_val=None) -> Tuple[int, pd.DataFrame]:
    """Pick the number of hidden states on VALIDATION data, never on test.

    Scored by per-observation held-out log-likelihood, so different K compare
    directly. The true activity count is not used.

    The score does not fall away again after its peak: it climbs, flattens off
    around K=7 and then stays level out to K=20, with the remaining wobble far
    below what the fit can resolve. Taking the plain argmax therefore picks an
    arbitrary K from the flat region — it returned 15 on this data, purely on
    noise. So instead we take the SMALLEST K that comes within SELECT_TOL of the
    best score, which lands on the point where the curve levels off and is
    stable across noise levels.
    """
    n_val = int(mask_val.sum()) if mask_val is not None else obs_val.size
    rows = []
    for K in candidates:
        pi, A, mu, sigma, _ = fit_hmm(obs_tr, K, seed, restarts=SEARCH_RESTARTS, mask=mask_tr)
        rows.append({"K": K,
                     "val_loglik_per_obs": loglikelihood(obs_val, pi, A, mu, sigma, mask_val) / n_val})
    table = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

    best = table["val_loglik_per_obs"].max()
    good = table[table["val_loglik_per_obs"] >= best - SELECT_TOL]
    return int(good["K"].iloc[0]), table


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(noise_std, sequences, weights, activities, n_cases,
             k_candidates: List[int]) -> Dict[str, float]:
    """Fit everything on train, choose K on validation, report on test.

    Note on the split: cases here are independent draws, so there is no time
    order to respect and the three slices are just disjoint case sets. A
    chronological split matters for the aggregated-signal experiment, not here.
    """
    obs, clean, labels, mask = simulate_cases(sequences, weights, activities,
                                              n_cases, noise_std, DATA_SEED)
    n_act, T = len(activities), obs.shape[1]
    c1 = int(round(n_cases * TRAIN_FRACTION))
    c2 = c1 + int(round(n_cases * VAL_FRACTION))
    tr, va, te = slice(0, c1), slice(c1, c2), slice(c2, n_cases)

    # Padded slots are excluded from every statistic and every score.
    m_tr, m_va, m_te = mask[tr], mask[va], mask[te]
    lab_tr, lab_te = labels[tr][m_tr], labels[te][m_te]
    obs_tr_flat = obs[tr][m_tr]

    # Reference energies come from TRAINING cases only.
    mu_true_te = np.array([clean[te][m_te & (labels[te] == a)].mean() for a in range(n_act)])

    # 1. Linear regression baseline — labels known, one-hot design → per-activity means
    reg_mu = np.linalg.lstsq(np.eye(n_act)[lab_tr], obs_tr_flat, rcond=None)[0]

    # 2. Position baseline — most common activity at each step number, from train
    pos_pred_per_t = np.array([
        np.bincount(labels[tr][:, t][m_tr[:, t]], minlength=n_act).argmax()
        if m_tr[:, t].any() else 0
        for t in range(T)])
    pos_acc = float((np.tile(pos_pred_per_t, (te.stop - te.start, 1))[m_te] == lab_te).mean())

    # 3. HMM — labels hidden, order modelled, K chosen on validation
    K, k_table = select_k(obs[tr], obs[va], k_candidates, DATA_SEED, m_tr, m_va)
    pi, A, hmm_mu, hmm_sigma, _ = fit_hmm(obs[tr], K, seed=DATA_SEED, mask=m_tr)
    hmm_tr_pred = viterbi(obs[tr], pi, A, hmm_mu, hmm_sigma, m_tr)[m_tr]
    hmm_map = map_states_from_train(hmm_tr_pred, lab_tr, K, n_act)
    hmm_pred = hmm_map[viterbi(obs[te], pi, A, hmm_mu, hmm_sigma, m_te)[m_te]]
    hmm_energy = energy_per_activity(hmm_mu, hmm_map, hmm_tr_pred, n_act)

    # 4. Gaussian mixture — labels hidden, order ignored, same K
    gmm = GaussianMixture(n_components=K, n_init=N_RESTARTS, random_state=DATA_SEED)
    gmm.fit(obs_tr_flat.reshape(-1, 1))
    gmm_tr_pred = gmm.predict(obs_tr_flat.reshape(-1, 1))
    gmm_map = map_states_from_train(gmm_tr_pred, lab_tr, K, n_act)
    gmm_pred = gmm_map[gmm.predict(obs[te][m_te].reshape(-1, 1))]
    gmm_energy = energy_per_activity(gmm.means_.ravel(), gmm_map, gmm_tr_pred, n_act)

    def err(est: np.ndarray) -> float:
        """Worst per-activity energy error. Activities no state claimed count as
        misses, not as free passes."""
        if np.isnan(est).any():
            return float("inf")
        return float(np.abs(est - mu_true_te).max())

    return {
        "noise_std":     noise_std,
        "chosen_K":      K,
        "reg_max_err":   float(np.abs(reg_mu - mu_true_te).max()),
        "hmm_max_err":   err(hmm_energy),
        "gmm_max_err":   err(gmm_energy),
        "hmm_state_acc": float((hmm_pred == lab_te).mean()),
        "pos_state_acc": pos_acc,
        "gmm_state_acc": float((gmm_pred == lab_te).mean()),
        "_k_table":      k_table,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(k: int, wanted: List[int], n_cases: int, noise: float | None) -> None:
    out_dir = Path(__file__).resolve().parent / f"generated_signals_k{k}"
    sequences, weights, activities = load_variants(out_dir, wanted)

    print(f"Variants used: {', '.join(str(v) for v in wanted)}")
    for v, seq, w in zip(wanted, sequences, weights):
        print(f"\n  variant {v}  ({w:.1%} of cases)")
        for i, a in enumerate(seq, 1):
            print(f"    {i}. {a:<30} base cost {BASE_COSTS[a]:g}")

    n_tr = int(round(n_cases * TRAIN_FRACTION))
    n_va = int(round(n_cases * VAL_FRACTION))
    print(f"\nTrue activity count: {len(activities)} (NOT given to the HMM)")
    print(f"K searched over {K_CANDIDATES}, chosen on validation log-likelihood")
    print(f"{n_cases} cases: {n_tr} train / {n_va} validation / {n_cases - n_tr - n_va} test\n")

    levels = [noise] if noise is not None else NOISE_SWEEP
    results = [evaluate(s, sequences, weights, activities, n_cases, K_CANDIDATES)
               for s in levels]

    print("number of hidden states chosen on validation (per-observation log-likelihood)")
    for r in results:
        row = " ".join(f"K={int(t.K)}:{t.val_loglik_per_obs:+.3f}"
                       for t in r["_k_table"].itertuples())
        print(f"  noise {r['noise_std']:>5}: {row}   → chose K={r['chosen_K']}")
    print()

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_k_table"} for r in results])

    print("=" * 96)
    print("energy recovery — largest error across activities (lower is better)")
    print("=" * 96)
    print(df[["noise_std", "chosen_K", "reg_max_err", "hmm_max_err", "gmm_max_err"]]
          .rename(columns={"reg_max_err": "Lin.reg. baseline (labels known)",
                           "hmm_max_err": "HMM (hidden + order)",
                           "gmm_max_err": "Mixture (hidden, no order)"})
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n" + "=" * 96)
    print("state recovery — share of steps assigned to the right activity")
    print("=" * 96)
    print(df[["noise_std", "hmm_state_acc", "pos_state_acc", "gmm_state_acc"]]
          .rename(columns={"hmm_state_acc": "HMM (hidden + order)",
                           "pos_state_acc": "Step number only",
                           "gmm_state_acc": "Mixture (no order)"})
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    tag = "".join(str(v) for v in wanted)
    path = out_dir / f"hmm_variants{tag}_comparison.csv"
    df.to_csv(path, index=False)
    print(f"\nsaved → {out_dir.name}/{path.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HMM across process variants vs linear regression baseline.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--variants", type=str, default="1,2,3", help="e.g. 1, 1,2 or 1,2,3")
    p.add_argument("--n-cases", type=int, default=4000)
    p.add_argument("--noise", type=float, default=None)
    a = p.parse_args()
    main(a.k, [int(v) for v in a.variants.split(",")], a.n_cases, a.noise)
