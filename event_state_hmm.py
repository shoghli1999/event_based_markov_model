"""One consistent event-state HMM for the thesis experiments.

The meaning of a state never changes in this file:

    hidden state = process activity
    transition   = activity -> next activity
    reward       = energy consumed by that activity

The observed facility signal is still aggregated into 30-minute intervals.  The
main attribution test gives every method the same logged activity counts.  A
separate, harder decoder test hides activity labels after the split and uses the
aggregate meter residual plus learned transitions to reconstruct each case path.

The implementation has two forms of the same model:

* pooled: one event-state HMM for every retained case;
* variant-first: deterministic event-state HMM experts, one per training path.

Both use the same activity states and Gaussian activity-energy rewards.  There
are no background states.  The known synthetic baseline is subtracted for both
the HMM and regression reference, exactly as in the original controlled model.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import nnls

from data_pipeline import (
    FREQ,
    SIGNAL_SEED,
    background_noise,
    _duration_share,
    _spread_signal_and_matrix,
    sine_baseline,
    base_cost_table,
    build_signal_and_matrix,
    filter_learnable_events,
    load_events,
    select_coverage,
    simulate,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_event_state"
TRAIN_FRACTION = 0.70
TRANSITION_SMOOTHING = 1e-10
MIN_SIGMA = 0.50
MIN_GAP_SIGMA = 0.25
MIN_VARIANCE = 1e-8
VARIANCE_FIT_ITERS = 2
BAUM_WELCH_ITERS = 2
# The log names the activity of every training event, so no state is hidden
# while the emissions are learned.  Baum-Welch is kept and can be switched on,
# but it is off by default because it measurably loses accuracy; the evidence
# is produced by final_model/baum_welch_check.py.
BAUM_WELCH_DEFAULT = False
VITERBI_SWEEPS = 3
# Zero means that the whole-log experiment keeps every training variant.
FULL_LOG_EXPERTS_PER_LENGTH = 0
RQ2_SPANS = [10080.0, 120.0, 30.0, 0.0]
RIDGE_ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
# Thirty consecutive seeds make the repeat easy to reproduce and reduce the
# chance that a conclusion depends on a few fortunate noise draws.
ROBUSTNESS_SEEDS = list(range(30))
DECODER_SEEDS = [0, 1, 2, 3, 4]
RQ2_SEEDS = [0, 1, 2, 3, 4]


@dataclass
class Problem:
    """All aligned inputs for one experimental scope."""

    scope: str
    events: pd.DataFrame
    activities: list[str]
    X: np.ndarray
    event_cost: np.ndarray
    target: np.ndarray
    truth: np.ndarray
    cut: int
    complete_log_events: int
    # Fraction of each event that belongs to the following interval.  Anything
    # that rebuilds interval counts must split events the same way the signal
    # was built, or it is scored against energy its counts never claim.
    share_next: np.ndarray | None = None
    # The daily background shape.  Noise is larger at the hours when it is high,
    # so the variance model needs it alongside the number of events.
    background_shape: np.ndarray | None = None


@dataclass
class EventStateHMM:
    """An HMM whose state indices map directly to activity names."""

    activities: list[str]
    start: np.ndarray
    transition: np.ndarray
    weighted_energy: np.ndarray
    energy: np.ndarray
    sigma: np.ndarray
    gap_mean: np.ndarray
    gap_sigma: np.ndarray
    interval_noise_variance: float
    event_variance: float
    background_variance: float
    baum_welch_used: bool


def whole_event_counts(problem: Problem) -> np.ndarray:
    """Interval counts with each event charged whole to the interval it starts in.

    The model shares an event across the two intervals its duration covers, which
    is right for the energy.  For reporting rank it is wrong: two activities that
    always occur together end up with columns that differ by a hair, purely
    because each event carries a random duration, and the rank then over-reports
    what the data can tell apart.  Counting whole events measures the process.
    """
    bins = problem.events["bin"].to_numpy(int)
    states = problem.events["state"].to_numpy(int)
    counts = np.zeros((len(problem.target), len(problem.activities)))
    np.add.at(counts, (bins, states), 1.0)
    return counts


def training_rank(problem: Problem) -> int:
    """How many activity costs the training data can tell apart."""
    return int(np.linalg.matrix_rank(whole_event_counts(problem)[: problem.cut]))


def event_count_split(X: np.ndarray, fraction: float = TRAIN_FRACTION) -> int:
    """Chronological cut after ``fraction`` of all events have occurred."""
    per_interval = X.sum(axis=1)
    cut = int(np.searchsorted(np.cumsum(per_interval), fraction * per_interval.sum()) + 1)
    return int(np.clip(cut, 10, len(X) - 10))


@lru_cache(maxsize=1)
def _simulated_complete_events() -> pd.DataFrame:
    """Load and simulate the large log once per program run."""
    events = load_events()
    activities = sorted(events["activity"].unique())
    return simulate(events, base_cost_table(activities))


def load_problem(scope: str, spread_duration: bool = True) -> Problem:
    """Create a deterministic signal for top-N variants or the learnable log.

    Events are aggregated with my supervisor's convention: one that crosses a
    30-minute boundary has its energy and its activity count shared between the
    two intervals. ``spread_duration=False`` exists only for
    final_model/interval_convention_check.py.
    """
    # Scope functions return copies, so the cached complete table stays intact.
    all_events = _simulated_complete_events()
    complete_log_events = len(all_events)
    if scope == "learnable":
        events, _ = filter_learnable_events(all_events)
    elif scope.startswith("top"):
        events = select_coverage(all_events, scope.removeprefix("top"))
    else:
        raise ValueError(f"Unknown scope: {scope}")

    activities = sorted(events["activity"].unique())
    signal, matrix = build_signal_and_matrix(events, spread_duration=spread_duration)
    timeline = pd.DatetimeIndex(pd.to_datetime(matrix["timestamp"]))

    events = events.sort_values(["case", "ts"]).reset_index(drop=True)
    events["bin"] = timeline.get_indexer(events["ts"].dt.floor(FREQ))
    gap_minutes = events.groupby("case", sort=False)["ts"].diff().dt.total_seconds()
    events["gap"] = np.log1p(gap_minutes.fillna(0.0).clip(lower=0.0) / 60.0)
    state_index = {activity: index for index, activity in enumerate(activities)}
    events["state"] = events["activity"].map(state_index).astype(int)

    # The exact synthetic background is removed for both methods.  This is an
    # oracle experiment and is reported as such in every result table.
    event_cost = signal["event_cost"].to_numpy(float)
    target = event_cost + signal["noise"].to_numpy(float)
    X = matrix[activities].to_numpy(float)
    share_next = (
        _duration_share(events, events["ts"].dt.floor(FREQ))
        if spread_duration
        else np.zeros(len(events))
    )
    truth = (
        events.groupby("activity")["energy"]
        .mean()
        .reindex(activities)
        .to_numpy(float)
    )
    return Problem(
        scope,
        events,
        activities,
        X,
        event_cost,
        target,
        truth,
        event_count_split(X),
        complete_log_events,
        share_next,
        signal["shape"].to_numpy(float),
    )


def with_noise(problem: Problem, seed: int) -> Problem:
    """Return the same simulated process with one independent noise draw."""
    target = problem.event_cost + background_noise(problem.background_shape, seed)
    return Problem(
        problem.scope,
        problem.events,
        problem.activities,
        problem.X,
        problem.event_cost,
        target,
        problem.truth,
        problem.cut,
        problem.complete_log_events,
        problem.share_next,
        problem.background_shape,
    )


def _compress_case_times(events: pd.DataFrame, span_minutes: float) -> pd.Series:
    """Move each case's events into a shorter span without changing their order."""
    first = events.groupby("case")["ts"].transform("min")
    offset = events["ts"] - first
    own_span = events.groupby("case")["ts"].transform("max") - first
    target = pd.Timedelta(minutes=span_minutes)
    safe_span = own_span.replace(pd.Timedelta(0), pd.Timedelta(seconds=1))
    factor = np.where(own_span > target, target / safe_span, 1.0)
    return first + offset * factor


def compressed_problem(
    base: Problem, span_minutes: float, noise_seed: int = SIGNAL_SEED
) -> Problem:
    """Re-bin one event set so RQ2 can control activity-column correlation."""
    events = base.events.copy()
    stamps = _compress_case_times(events, span_minutes)
    events["analysis_ts"] = stamps
    gap_minutes = (
        events.groupby("case", sort=False)["analysis_ts"].diff().dt.total_seconds()
    )
    events["gap"] = np.log1p(gap_minutes.fillna(0.0).clip(lower=0.0) / 60.0)
    events = events.drop(columns="analysis_ts")
    interval = stamps.dt.floor(FREQ)
    timeline = pd.date_range(interval.min(), interval.max(), freq=FREQ)
    activities = base.activities

    # The compressed timeline uses the same sharing as the main experiment,
    # otherwise RQ2 would silently test a different model.
    compressed = events.assign(ts=stamps)
    event_cost, matrix = _spread_signal_and_matrix(
        compressed, interval, timeline, activities
    )
    share_next = _duration_share(compressed, interval)
    shape = sine_baseline(timeline)
    noise = background_noise(shape, noise_seed)
    events["bin"] = timeline.get_indexer(interval)
    X = matrix[activities].to_numpy(float)
    return Problem(
        f"rq2_{span_minutes:g}min",
        events,
        activities,
        X,
        event_cost,
        event_cost + noise,
        base.truth,
        event_count_split(X),
        base.complete_log_events,
        share_next,
        shape,
    )


def _complete_training_cases(events: pd.DataFrame, cut: int) -> set[str]:
    """Cases whose complete path is available before the test period."""
    last_bin = events.groupby("case", sort=False)["bin"].max()
    return set(last_bin[last_bin < cut].index)


def _transition_parameters(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Learn start and activity-transition probabilities from training cases."""
    K = len(problem.activities)
    start = np.full(K, TRANSITION_SMOOTHING)
    transition = np.full((K, K), TRANSITION_SMOOTHING)
    complete = _complete_training_cases(problem.events, problem.cut)

    for _, case in problem.events[problem.events["case"].isin(complete)].groupby(
        "case", sort=False
    ):
        states = case["state"].to_numpy(int)
        if not len(states):
            continue
        start[states[0]] += 1.0
        np.add.at(transition, (states[:-1], states[1:]), 1.0)

    start /= start.sum()
    transition /= transition.sum(axis=1, keepdims=True)
    return start, transition


def _fit_rewards(problem: Problem) -> np.ndarray:
    """OLS activity rewards from the aggregate training signal."""
    return np.linalg.lstsq(
        problem.X[: problem.cut], problem.target[: problem.cut], rcond=None
    )[0]


def weighted_fit(
    X: np.ndarray, y: np.ndarray, background: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feasible generalized least squares on interval totals.

    An interval can be noisy for two reasons, and the variance model has one term
    for each.  It holds many events, and every event brings its own unpredictable
    duration.  Or it falls at an hour when the building background is high, and
    the background swings in proportion to its own level.  So the variance is
    modelled as

        a + b * (events in the interval) + c * (background shape) ** 2

    with a, b and c learned from the training residuals, never from the
    generator.  The activity costs are then refitted with that weighting.
    Variances cannot be negative, so the terms come from NNLS; clipping a
    negative least-squares value would leave the others biased.

    Returns the activity costs, the variance terms, and each interval's variance.
    """
    columns = [np.ones(len(X)), X.sum(axis=1)]
    if background is not None:
        columns.append(np.asarray(background, dtype=float) ** 2)
    design = np.column_stack(columns)
    energy = np.linalg.lstsq(X, y, rcond=None)[0]
    parameters = np.ones(design.shape[1])
    variance = np.ones(len(X))

    for _ in range(VARIANCE_FIT_ITERS):
        parameters = np.maximum(nnls(design, (y - X @ energy) ** 2)[0], MIN_VARIANCE)
        variance = np.maximum(design @ parameters, MIN_VARIANCE)
        scale = 1.0 / np.sqrt(variance)
        energy = np.linalg.lstsq(X * scale[:, None], y * scale, rcond=None)[0]

    return energy, parameters, variance


def _fit_aggregate_emissions(
    problem: Problem,
) -> tuple[np.ndarray, float, float, float]:
    """Activity costs and the three variance terms, from training intervals only."""
    background = (
        None if problem.background_shape is None
        else problem.background_shape[: problem.cut]
    )
    energy, parameters, _ = weighted_fit(
        problem.X[: problem.cut], problem.target[: problem.cut], background
    )
    third = float(parameters[2]) if len(parameters) > 2 else 0.0
    return energy, float(parameters[0]), float(parameters[1]), third


def _fit_ridge(problem: Problem) -> tuple[np.ndarray, float]:
    """Choose a ridge penalty on training validation data, then refit."""
    inner_cut = event_count_split(problem.X[: problem.cut], 0.80)
    X_train = problem.X[:inner_cut]
    y_train = problem.target[:inner_cut]
    X_valid = problem.X[inner_cut : problem.cut]
    y_valid = problem.target[inner_cut : problem.cut]
    identity = np.eye(problem.X.shape[1])
    best_alpha, best_error = 0.0, np.inf
    for alpha in RIDGE_ALPHAS:
        if alpha == 0.0:
            weights = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        else:
            weights = np.linalg.solve(
                X_train.T @ X_train + alpha * identity, X_train.T @ y_train
            )
        error = float(np.mean((y_valid - X_valid @ weights) ** 2))
        if error < best_error:
            best_alpha, best_error = alpha, error

    X_train = problem.X[: problem.cut]
    y_train = problem.target[: problem.cut]
    if best_alpha == 0.0:
        weights = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    else:
        weights = np.linalg.solve(
            X_train.T @ X_train + best_alpha * identity, X_train.T @ y_train
        )
    return weights, best_alpha


def _timing_parameters(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Estimate activity-specific event-gap emissions from complete training cases."""
    complete = _complete_training_cases(problem.events, problem.cut)
    training = problem.events[problem.events["case"].isin(complete)]
    K = len(problem.activities)
    means = np.zeros(K)
    sigmas = np.ones(K)
    for state in range(K):
        values = training.loc[training["state"] == state, "gap"].to_numpy(float)
        if len(values):
            means[state] = float(values.mean())
        if len(values) > 1:
            sigmas[state] = max(float(values.std()), MIN_GAP_SIGMA)
    return means, sigmas


def _training_observations(problem: Problem, energy: np.ndarray) -> pd.Series:
    """Share only the unexplained training residual between simultaneous events.

    The activity reward comes from the aggregate Gaussian emission fit.  The
    residual share is used as a noisy event-level observation for Baum-Welch;
    generated per-event ground-truth energy is never read here.
    """
    prediction = problem.X @ energy
    residual = problem.target - prediction
    counts = np.maximum(problem.X.sum(axis=1), 1.0)
    bins = problem.events["bin"].to_numpy(int)
    states = problem.events["state"].to_numpy(int)
    values = energy[states] + residual[bins] / counts[bins]
    return pd.Series(values, index=problem.events.index)


def _pack_cases(events: pd.DataFrame, values: pd.Series, cases: set[str]):
    """Flatten complete cases for hmmlearn and return their lengths."""
    pieces, lengths = [], []
    selected = events[events["case"].isin(cases)]
    for _, case in selected.groupby("case", sort=False):
        pieces.append(values.loc[case.index].to_numpy(float))
        lengths.append(len(case))
    return (np.concatenate(pieces) if pieces else np.empty(0), lengths)


def _reward_estimator_name(model: EventStateHMM) -> str:
    """Name the estimator that actually produced the rewards in this run."""
    return (
        "baum_welch_from_weighted_initialisation"
        if model.baum_welch_used
        else "feasible_generalized_least_squares"
    )


def fit_pooled_hmm(
    problem: Problem,
    use_transitions: bool = True,
    baum_welch: bool = BAUM_WELCH_DEFAULT,
) -> EventStateHMM:
    """Fit an activity-state HMM using training data only.

    Activity labels define the semantic state names, as in the original model.
    Activity rewards come from the weighted regression on interval totals,
    which is the only estimator that reads the measured signal directly.

    Baum-Welch is kept and can be switched on, but it is off by default.  The
    log already names the activity of every training event, so nothing about
    the hidden states is unknown at training time; running Baum-Welch there
    discards labels that were already correct and it has to invent an
    event-level value from each interval's leftover.  On all five scopes that
    loses accuracy, so the Markov chain is used where the states really are
    hidden instead: Viterbi decoding of the test period and the reward layer.
    """
    start, transition = _transition_parameters(problem)
    if not use_transitions:
        K = len(problem.activities)
        start = np.full(K, 1.0 / K)
        transition = np.full((K, K), 1.0 / K)
    gap_mean, gap_sigma = _timing_parameters(problem)
    initial_energy, interval_noise_variance, event_variance, background_variance = (
        _fit_aggregate_emissions(problem)
    )
    observations = _training_observations(problem, initial_energy)
    complete = _complete_training_cases(problem.events, problem.cut)
    packed, lengths = _pack_cases(problem.events, observations, complete)
    K = len(problem.activities)

    train_events = problem.events[
        (problem.events["bin"] < problem.cut)
        & problem.events["case"].isin(complete)
    ]
    state = train_events["state"].to_numpy(int)
    obs = observations.loc[train_events.index].to_numpy(float)
    sigma = np.full(K, MIN_SIGMA)
    for k in range(K):
        values = obs[state == k]
        if len(values) > 1:
            sigma[k] = max(float(values.std()), MIN_SIGMA)

    model = GaussianHMM(
        n_components=K,
        covariance_type="diag",
        n_iter=BAUM_WELCH_ITERS,
        tol=1e-5,
        min_covar=MIN_SIGMA**2,
        init_params="",
        params="mc",
        implementation="log",
        random_state=42,
    )
    model.startprob_ = start
    model.transmat_ = transition
    model.means_ = initial_energy[:, None]
    model.covars_ = (sigma**2)[:, None]

    used = False
    if baum_welch and len(packed):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(packed[:, None], lengths)
                candidate_mean = np.asarray(model.means_)
                candidate_variance = np.asarray(model.covars_)
                history = np.asarray(model.monitor_.history, dtype=float)
                monotone = len(history) < 2 or np.all(np.diff(history) >= -1e-6)
                used = bool(
                    np.all(np.isfinite(candidate_mean))
                    and np.all(np.isfinite(candidate_variance))
                    and monotone
                )
            except (ValueError, np.linalg.LinAlgError):
                used = False

    energy = (
        np.asarray(model.means_).reshape(K, -1)[:, 0]
        if used else initial_energy
    )
    learned_sigma = (
        np.sqrt(np.asarray(model.covars_).reshape(K, -1)[:, 0]) if used else sigma
    )

    # If the event-level update is switched off or fails, keep the weighted
    # reward and the training-only variance, and record that in the saved
    # result instead of hiding it.
    if not used:
        energy, learned_sigma, used = initial_energy, sigma, False

    return EventStateHMM(
        problem.activities,
        start,
        transition,
        initial_energy,
        energy,
        learned_sigma,
        gap_mean,
        gap_sigma,
        interval_noise_variance,
        event_variance,
        background_variance,
        used,
    )


def _viterbi(
    observation: np.ndarray, gap: np.ndarray, model: EventStateHMM
) -> np.ndarray:
    """Most likely activity path for one case."""
    if not len(observation):
        return np.empty(0, dtype=np.int32)
    sigma = np.maximum(model.sigma, MIN_SIGMA)
    log_emission = (
        -0.5 * ((observation[:, None] - model.energy[None, :]) / sigma[None, :]) ** 2
        - np.log(sigma[None, :])
    )
    gap_sigma = np.maximum(model.gap_sigma, MIN_GAP_SIGMA)
    log_emission += (
        -0.5 * ((gap[:, None] - model.gap_mean[None, :]) / gap_sigma[None, :]) ** 2
        - np.log(gap_sigma[None, :])
    )
    log_A = np.log(np.maximum(model.transition, 1e-300))
    score = np.log(np.maximum(model.start, 1e-300)) + log_emission[0]
    back = np.zeros((len(observation), len(model.energy)), dtype=np.int32)
    for step in range(1, len(observation)):
        candidates = score[:, None] + log_A
        back[step] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_emission[step]
    path = np.empty(len(observation), dtype=np.int32)
    path[-1] = int(score.argmax())
    for step in range(len(observation) - 2, -1, -1):
        path[step] = back[step + 1, path[step + 1]]
    return path


def _position_initialisation(problem: Problem) -> np.ndarray:
    """Initial test activities learned from training case length and position."""
    events = problem.events.copy()
    events["length"] = events.groupby("case")["case"].transform("size")
    events["position"] = events.groupby("case").cumcount()
    complete = _complete_training_cases(events, problem.cut)
    training = events[events["case"].isin(complete)]
    K = len(problem.activities)
    fallback = int(np.bincount(training["state"], minlength=K).argmax())
    lookup = {
        key: int(np.bincount(group["state"], minlength=K).argmax())
        for key, group in training.groupby(["length", "position"])
    }
    return np.array(
        [lookup.get((length, position), fallback)
         for length, position in zip(events["length"], events["position"])],
        dtype=np.int32,
    )


def _decoded_counts(
    states: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    n_states: int,
    share_next: np.ndarray | None = None,
) -> np.ndarray:
    """Interval counts implied by a decoded state path.

    ``share_next`` is the fraction of each event that belongs to the following
    interval.  It must match the convention the signal was built with, otherwise
    the reconstruction is compared against energy the counts never claim, and the
    aggregate error becomes meaningless rather than merely worse.
    """
    if share_next is None:
        share_next = np.zeros(len(bins))
    here = bins.astype(np.int64)
    state = states.astype(np.int64)
    counts = np.zeros((n_bins, n_states))
    np.add.at(counts, (here, state), 1.0 - share_next)
    following = here + 1
    inside = following < n_bins
    np.add.at(counts, (following[inside], state[inside]), share_next[inside])
    return counts


def _event_share(problem: Problem) -> np.ndarray:
    """Per-event share belonging to the next interval; zeros when not shared."""
    if problem.share_next is None:
        return np.zeros(len(problem.events))
    return np.asarray(problem.share_next, dtype=float)


def _apply_energy(
    prediction: np.ndarray,
    bins: np.ndarray,
    values: np.ndarray,
    share_next: np.ndarray,
) -> None:
    """Add each event's energy to its interval and the next, split by share."""
    np.add.at(prediction, bins, values * (1.0 - share_next))
    following = bins + 1
    inside = following < len(prediction)
    np.add.at(prediction, following[inside], (values * share_next)[inside])


def decode_pooled(problem: Problem, model: EventStateHMM):
    """Coordinate Viterbi over all concurrent activity-state chains.

    Cases overlap in the meter.  One case is decoded at a time after subtracting
    the current contribution of every other case.  This is retrospective and
    approximate; exact joint decoding would grow exponentially with the number
    of simultaneous cases.
    """
    events = problem.events
    bins = events["bin"].to_numpy(int)
    gaps = events["gap"].to_numpy(float)
    share = _event_share(problem)
    test = bins >= problem.cut
    states = _position_initialisation(problem)
    rows_by_case = [
        group.index.to_numpy()[bins[group.index] >= problem.cut]
        for _, group in events.groupby("case", sort=False)
        if np.any(bins[group.index] >= problem.cut)
    ]
    rows_by_case = [rows for rows in rows_by_case if len(rows)]

    counts = _decoded_counts(
        states[test], bins[test], len(problem.target), len(model.energy), share[test]
    )
    prediction = counts @ model.energy
    history = []

    for sweep in range(1, VITERBI_SWEEPS + 1):
        changes = 0
        for rows in rows_by_case:
            old = states[rows]
            event_bins = bins[rows]
            _apply_energy(prediction, event_bins, -model.energy[old], share[rows])
            _, inverse, multiplicity = np.unique(
                event_bins, return_inverse=True, return_counts=True
            )
            observation = (
                problem.target[event_bins] - prediction[event_bins]
            ) / multiplicity[inverse]
            new = _viterbi(observation, gaps[rows], model)
            changes += int(np.any(new != old))
            states[rows] = new
            _apply_energy(prediction, event_bins, model.energy[new], share[rows])

        counts = _decoded_counts(
            states[test], bins[test], len(problem.target), len(model.energy),
            share[test],
        )
        prediction = counts @ model.energy
        error = float(np.sqrt(np.mean(
            (problem.target[problem.cut :] - prediction[problem.cut :]) ** 2
        )))
        history.append({"sweep": sweep, "test_rmse": error, "changed_cases": changes})
        if changes == 0:
            break

    return states, counts, pd.DataFrame(history)


def decode_without_transitions(problem: Problem, model: EventStateHMM):
    """Matched decoder control with the Markov start and transitions removed.

    Energy emissions, timing emissions, initialization, residual sharing and
    coordinate sweeps stay unchanged.  Therefore a difference from
    ``decode_pooled`` is evidence about the sequence probabilities themselves.
    """
    K = len(model.activities)
    control = replace(
        model,
        start=np.full(K, 1.0 / K),
        transition=np.full((K, K), 1.0 / K),
    )
    return decode_pooled(problem, control)


def _training_templates(problem: Problem, limit_per_length: int):
    """Training paths by case length; a zero limit keeps every path."""
    complete = _complete_training_cases(problem.events, problem.cut)
    all_counts = (
        problem.events[problem.events["case"].isin(complete)]
        .groupby("case", sort=False)["state"]
        .apply(tuple)
        .value_counts()
    )
    by_length: dict[int, list[tuple[np.ndarray, int]]] = defaultdict(list)
    for path, frequency in all_counts.items():
        if limit_per_length == 0 or len(by_length[len(path)]) < limit_per_length:
            by_length[len(path)].append(
                (np.asarray(path, dtype=np.int32), int(frequency))
            )
    selected = sum(len(paths) for paths in by_length.values())
    return by_length, len(all_counts), selected


def _seen_path_coverage(
    problem: Problem,
    templates: dict[int, list[tuple[np.ndarray, int]]],
) -> float:
    """Score held-out path coverage without changing any decoded state.

    This is the only variant helper that reads the complete held-out activity
    path. It runs for evaluation after candidate paths have already been built;
    its result cannot affect path selection.
    """
    events = problem.events
    bins = events["bin"].to_numpy(int)
    logged_states = events["state"].to_numpy(int)
    test = bins >= problem.cut
    represented_events = 0
    total_test_events = int(test.sum())

    for _, case in events.groupby("case", sort=False):
        rows = case.index.to_numpy()
        positions = np.arange(len(rows))
        test_positions = positions[test[rows]]
        if not len(test_positions):
            continue
        candidates = templates.get(len(rows), [])
        known_positions = positions[~test[rows]]
        if len(known_positions):
            candidates = [
                item for item in candidates
                if np.array_equal(
                    item[0][known_positions],
                    logged_states[rows[known_positions]],
                )
            ] or candidates
        if any(
            np.array_equal(path, logged_states[rows])
            for path, _ in candidates
        ):
            represented_events += len(test_positions)

    return 100.0 * represented_events / max(total_test_events, 1)


def decode_variant_first(problem: Problem, model: EventStateHMM, experts: int):
    """Decode with frequent deterministic event-state HMM experts.

    The true test variant is never read.  If no pretrained path fits a case, the
    same pooled event-state HMM supplies a fallback path.

    Every energy calculation here splits an event across the two intervals its
    duration covers: the running reconstruction, the reported counts and the
    candidate scores alike.
    """
    events = problem.events
    bins = events["bin"].to_numpy(int)
    gaps = events["gap"].to_numpy(float)
    share = _event_share(problem)
    test = bins >= problem.cut
    interval_event_count = np.bincount(bins, minlength=len(problem.target))
    background_level = (
        np.zeros(len(problem.target)) if problem.background_shape is None
        else problem.background_shape
    )
    templates, available, selected = _training_templates(problem, experts)
    states = _position_initialisation(problem)
    jobs = []

    for _, case in events.groupby("case", sort=False):
        rows = case.index.to_numpy()
        positions = np.arange(len(rows))
        test_positions = positions[test[rows]]
        if not len(test_positions):
            continue
        candidates = templates.get(len(rows), [])
        known_positions = positions[~test[rows]]
        if len(known_positions):
            # These labels are before the cut. Held-out positions are never
            # indexed here.
            observed_training_states = case["state"].to_numpy(int)[known_positions]
            compatible = [
                item for item in candidates
                if np.array_equal(item[0][known_positions], observed_training_states)
            ]
            if compatible:
                candidates = compatible
        if candidates:
            states[rows[test_positions]] = candidates[0][0][test_positions]
            candidate_paths = np.stack([path for path, _ in candidates])
            frequencies = np.asarray([frequency for _, frequency in candidates])
        else:
            candidate_paths = np.empty((0, len(rows)), dtype=np.int32)
            frequencies = np.empty(0)
        jobs.append(
            (rows[test_positions], test_positions, candidate_paths, frequencies)
        )

    prediction = np.zeros(len(problem.target))
    _apply_energy(prediction, bins[test], model.energy[states[test]], share[test])
    history = []

    for sweep in range(1, VITERBI_SWEEPS + 1):
        changes = 0
        for rows, positions, candidate_paths, frequencies in jobs:
            old = states[rows]
            event_bins = bins[rows]
            _apply_energy(prediction, event_bins, -model.energy[old], share[rows])
            unique_bins, inverse, multiplicity = np.unique(
                event_bins, return_inverse=True, return_counts=True
            )
            remaining = problem.target[unique_bins] - prediction[unique_bins]

            if len(candidate_paths):
                proposed = candidate_paths[:, positions]
                # A candidate's energy is spread across the two intervals each
                # event covers, exactly as the signal and the running
                # reconstruction are.  Scoring it whole in the starting interval
                # would compare a split residual against an unsplit proposal, and
                # the size of that mismatch differs from candidate to candidate.
                case_share = share[rows]
                spills = (event_bins + 1 < len(problem.target)) & (case_share > 0.0)
                window = np.unique(
                    np.concatenate([event_bins, event_bins[spills] + 1])
                )
                here = np.searchsorted(window, event_bins)
                following = np.searchsorted(window, event_bins + 1)
                window_remaining = problem.target[window] - prediction[window]

                contribution = np.zeros((len(proposed), len(window)))
                candidate_rows = np.arange(len(proposed))
                for position in range(len(rows)):
                    energy = model.energy[proposed[:, position]]
                    fraction = case_share[position]
                    contribution[candidate_rows, here[position]] += energy * (
                        1.0 - fraction
                    )
                    if spills[position]:
                        contribution[candidate_rows, following[position]] += (
                            energy * fraction
                        )

                energy_score = np.sum(
                    (window_remaining[None, :] - contribution) ** 2
                    / np.maximum(
                        model.interval_noise_variance
                        + model.event_variance * interval_event_count[window]
                        + model.background_variance * background_level[window] ** 2,
                        MIN_VARIANCE,
                    )[None, :],
                    axis=1,
                )
                gap_sigma = np.maximum(model.gap_sigma[proposed], MIN_GAP_SIGMA)
                timing_score = np.sum(
                    ((gaps[rows][None, :] - model.gap_mean[proposed]) / gap_sigma) ** 2
                    + 2.0 * np.log(gap_sigma),
                    axis=1,
                )
                score = energy_score + timing_score - 2.0 * np.log(
                    np.maximum(frequencies, 1)
                )
                winner = proposed[int(score.argmin())]
            else:
                observation = remaining[inverse] / multiplicity[inverse]
                winner = _viterbi(observation, gaps[rows], model)

            changes += int(np.any(winner != old))
            states[rows] = winner
            _apply_energy(prediction, event_bins, model.energy[winner], share[rows])

        error = float(np.sqrt(np.mean(
            (problem.target[problem.cut :] - prediction[problem.cut :]) ** 2
        )))
        history.append({"sweep": sweep, "test_rmse": error, "changed_cases": changes})
        if changes == 0:
            break

    best_counts = _decoded_counts(
        states[test], bins[test], len(problem.target), len(model.energy), share[test]
    )
    coverage = _seen_path_coverage(problem, templates)
    return (
        states,
        best_counts,
        coverage,
        available,
        selected,
        pd.DataFrame(history),
    )


def _metrics(
    problem: Problem,
    model: EventStateHMM,
    states: np.ndarray,
    counts: np.ndarray,
) -> dict[str, float]:
    """Score one decoder on the held-out period.

    Three views of the same decoded path: how well it rebuilds the meter total,
    how often it names the right activity, and how close it gets to each single
    event's energy.
    """
    bins = problem.events["bin"].to_numpy(int)
    test_events = bins >= problem.cut
    prediction = counts @ model.energy
    predicted_event_energy = model.energy[states[test_events]]
    true_event_energy = problem.events.loc[test_events, "energy"].to_numpy(float)
    residual = problem.target[problem.cut :] - prediction[problem.cut :]
    measured = problem.target[problem.cut :]
    return {
        # Squaring makes one slip on a giant interval outweigh thousands of
        # small ones, and this log arrives in bursts, so the two fairer
        # measurements are reported beside it.
        "test_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "test_mae": float(np.mean(np.abs(residual))),
        "test_relative_error": float(
            np.sum(np.abs(residual)) / max(np.sum(np.abs(measured)), 1e-9)
        ),
        "state_accuracy": float(np.mean(
            states[test_events] == problem.events.loc[test_events, "state"].to_numpy(int)
        )),
        "event_energy_mae": float(np.mean(
            np.abs(predicted_event_energy - true_event_energy)
        )),
    }


def _prefixed(name: str, metrics: dict[str, float]) -> dict[str, float]:
    """Name one decoder's measurements, so every decoder reports the same set."""
    return {f"{name}_{key}": value for key, value in metrics.items()}


def _rmse(target: np.ndarray, prediction: np.ndarray, cut: int) -> float:
    """Held-out root mean squared reconstruction error."""
    return float(np.sqrt(np.mean((target[cut:] - prediction[cut:]) ** 2)))


def _mae(target: np.ndarray, prediction: np.ndarray, cut: int) -> float:
    """Held-out mean absolute reconstruction error, as the proposal asks for."""
    return float(np.mean(np.abs(target[cut:] - prediction[cut:])))


def evaluate_scope(scope: str, experts: int | None = None):
    """Run the fair labelled and label-hidden comparisons for one scope."""
    problem = load_problem(scope)
    started = time.perf_counter()
    model = fit_pooled_hmm(problem, use_transitions=True)
    hmm_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    independent_model = fit_pooled_hmm(problem, use_transitions=False)
    independent_fit_seconds = time.perf_counter() - started

    ols_energy = _fit_rewards(problem)
    ridge_energy, ridge_alpha = _fit_ridge(problem)

    # Labelled reward comparison.  The aggregate Gaussian reward update is
    # weighted regression, so it is named as such rather than presented as a
    # Markov advantage.
    ols_rmse = _rmse(problem.target, problem.X @ ols_energy, problem.cut)
    ridge_rmse = _rmse(problem.target, problem.X @ ridge_energy, problem.cut)
    weighted_rmse = _rmse(
        problem.target, problem.X @ model.weighted_energy, problem.cut
    )
    hmm_rmse = _rmse(problem.target, problem.X @ model.energy, problem.cut)
    # The proposal asks for MSE and MAE; RMSE is the square root of the first.
    ols_mae = _mae(problem.target, problem.X @ ols_energy, problem.cut)
    ridge_mae = _mae(problem.target, problem.X @ ridge_energy, problem.cut)
    weighted_mae = _mae(
        problem.target, problem.X @ model.weighted_energy, problem.cut
    )
    hmm_mae = _mae(problem.target, problem.X @ model.energy, problem.cut)

    # Extra capability test: activity names are hidden after the split.  The
    # position baseline and both HMM decoders receive the same case/timing data.
    bins = problem.events["bin"].to_numpy(int)
    test_events = bins >= problem.cut
    position_states = _position_initialisation(problem)
    position_counts = _decoded_counts(
        position_states[test_events],
        bins[test_events],
        len(problem.target),
        len(problem.activities),
        _event_share(problem)[test_events],
    )
    position = _metrics(problem, model, position_states, position_counts)

    started = time.perf_counter()
    independent_em_states, independent_em_counts, _ = decode_without_transitions(
        problem, independent_model
    )
    independent_em_decode_seconds = time.perf_counter() - started
    independent_em = _metrics(
        problem, independent_model, independent_em_states, independent_em_counts
    )

    # Conservative transition ablation: keep the HMM emissions unchanged and
    # remove only its start/transition probabilities.
    started = time.perf_counter()
    independent_states, independent_counts, independent_history = (
        decode_without_transitions(problem, model)
    )
    independent_decode_seconds = time.perf_counter() - started
    independent = _metrics(problem, model, independent_states, independent_counts)

    started = time.perf_counter()
    pooled_states, pooled_counts, pooled_history = decode_pooled(problem, model)
    pooled_decode_seconds = time.perf_counter() - started
    pooled = _metrics(problem, model, pooled_states, pooled_counts)

    expert_count = experts if experts is not None else (
        FULL_LOG_EXPERTS_PER_LENGTH
        if scope == "learnable"
        else int(scope.removeprefix("top"))
    )
    started = time.perf_counter()
    (
        variant_states,
        variant_counts,
        coverage,
        training_variants,
        selected_experts,
        variant_history,
    ) = decode_variant_first(problem, model, expert_count)
    variant_decode_seconds = time.perf_counter() - started
    variant = _metrics(problem, model, variant_states, variant_counts)
    variant_lrm_rmse = _rmse(
        problem.target, variant_counts @ ols_energy, problem.cut
    )
    variant_lrm_event_mae = float(np.mean(np.abs(
        ols_energy[variant_states[test_events]]
        - problem.events.loc[test_events, "energy"].to_numpy(float)
    )))

    result = pd.DataFrame([{
        "scope": scope,
        "events": len(problem.events),
        "complete_log_events": problem.complete_log_events,
        "retained_event_percent": 100.0 * len(problem.events) / problem.complete_log_events,
        "activities": len(problem.activities),
        "K": len(problem.activities),
        "matrix_rank": training_rank(problem),
        "test_events": int(problem.X[problem.cut :].sum()),
        "ols_known_x_test_rmse": ols_rmse,
        "ols_known_x_test_mae": ols_mae,
        "ridge_known_x_test_mae": ridge_mae,
        "weighted_regression_test_mae": weighted_mae,
        "hmm_known_x_test_mae": hmm_mae,
        "ols_mean_cost_error": float(np.mean(np.abs(ols_energy - problem.truth))),
        "ridge_known_x_test_rmse": ridge_rmse,
        "ridge_mean_cost_error": float(
            np.mean(np.abs(ridge_energy - problem.truth))
        ),
        "ridge_alpha": ridge_alpha,
        "weighted_regression_test_rmse": weighted_rmse,
        "weighted_regression_mean_cost_error": float(
            np.mean(np.abs(model.weighted_energy - problem.truth))
        ),
        "hmm_known_x_test_rmse": hmm_rmse,
        "hmm_mean_cost_error": float(
            np.mean(np.abs(model.energy - problem.truth))
        ),
        "independent_em_mean_cost_error": float(
            np.mean(np.abs(independent_model.energy - problem.truth))
        ),
        "weighted_interval_noise_variance": model.interval_noise_variance,
        "weighted_event_variance": model.event_variance,
        "weighted_background_variance": model.background_variance,
        **_prefixed("position", position),
        **_prefixed("independent_decoder", independent),
        **_prefixed("independent_em_decoder", independent_em),
        **_prefixed("pooled_decoder", pooled),
        **_prefixed("variant_decoder", variant),
        "variant_hmm_lrm_test_rmse": variant_lrm_rmse,
        "variant_hmm_lrm_event_energy_mae": variant_lrm_event_mae,
        "experts_per_case_length": expert_count,
        "selected_variant_experts": selected_experts,
        "training_variants_available": training_variants,
        "represented_test_events_percent": coverage,
        "weighted_reward_estimator": "feasible_generalized_least_squares",
        "hmm_reward_estimator": _reward_estimator_name(model),
        "reward_variance_fit_iterations": VARIANCE_FIT_ITERS,
        "weighted_regression_uses_transitions": False,
        "hmm_transitions_affect_reward_estimate": model.baum_welch_used,
        "baum_welch_update_used": model.baum_welch_used,
        "uses_true_baseline": True,
        "main_comparison_uses_logged_test_activities": True,
        "decoder_uses_test_activity_labels": False,
        "top_variants_define_scope_before_split": scope.startswith("top"),
        "hmm_fit_seconds": hmm_fit_seconds,
        "independent_fit_seconds": independent_fit_seconds,
        "independent_em_decode_seconds": independent_em_decode_seconds,
        "independent_decode_seconds": independent_decode_seconds,
        "pooled_decode_seconds": pooled_decode_seconds,
        "variant_decode_seconds": variant_decode_seconds,
    }])
    return result, independent_history, pooled_history, variant_history


def run_rq2(
    spans: list[float] = RQ2_SPANS,
    seeds: list[int] = RQ2_SEEDS,
) -> pd.DataFrame:
    """Run the proposal's correlated-chain stress test with event states.

    Squeezing cases changes only the activity-column correlation.  OLS and
    weighted regression measure reward identifiability.  The matched
    no-transition, pooled HMM and variant-first decoders measure whether Markov
    sequence information changes hidden-event attribution.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_problem("top3")
    rows = []
    for span in spans:
        for seed in seeds:
            print(f"\nRQ2 span {span:g} minutes, noise seed {seed}")
            problem = compressed_problem(base, span, seed)
            gram = problem.X[: problem.cut].T @ problem.X[: problem.cut]
            eigenvalues = np.linalg.eigvalsh(gram)
            rank = training_rank(problem)
            condition = (
                float(eigenvalues.max() / eigenvalues.min())
                if eigenvalues.min() > 1e-12 else float("inf")
            )

            model = fit_pooled_hmm(problem, use_transitions=True)
            independent_model = fit_pooled_hmm(
                problem, use_transitions=False
            )
            independent_em_states, independent_em_counts, _ = (
                decode_without_transitions(problem, independent_model)
            )
            independent_em = _metrics(
                problem,
                independent_model,
                independent_em_states,
                independent_em_counts,
            )
            independent_states, independent_counts, _ = decode_without_transitions(
                problem, model
            )
            independent = _metrics(problem, model, independent_states, independent_counts)
            pooled_states, pooled_counts, _ = decode_pooled(problem, model)
            pooled = _metrics(problem, model, pooled_states, pooled_counts)
            states, counts, coverage, _, _, _ = decode_variant_first(
                problem, model, 3
            )
            variant = _metrics(problem, model, states, counts)
            ols = _fit_rewards(problem)
            rows.append({
                "span_minutes": span,
                "noise_seed": seed,
                "training_rank": rank,
                "activities": len(problem.activities),
                "condition_number": condition,
                "ols_mean_cost_error": float(np.mean(np.abs(ols - problem.truth))),
                "weighted_regression_mean_cost_error": float(
                    np.mean(np.abs(model.weighted_energy - problem.truth))
                ),
                "hmm_mean_cost_error": float(
                    np.mean(np.abs(model.energy - problem.truth))
                ),
                "independent_em_mean_cost_error": float(
                    np.mean(np.abs(independent_model.energy - problem.truth))
                ),
                "ols_known_x_test_rmse": _rmse(
                    problem.target, problem.X @ ols, problem.cut
                ),
                "weighted_regression_test_rmse": _rmse(
                    problem.target, problem.X @ model.weighted_energy, problem.cut
                ),
                "hmm_known_x_test_rmse": _rmse(
                    problem.target, problem.X @ model.energy, problem.cut
                ),
                **_prefixed("independent_decoder", independent),
                **_prefixed("independent_em_decoder", independent_em),
                **_prefixed("pooled_decoder", pooled),
                **_prefixed("variant_decoder", variant),
                "represented_test_events_percent": coverage,
                "weighted_reward_estimator": "feasible_generalized_least_squares",
                "hmm_reward_estimator": _reward_estimator_name(model),
                "weighted_regression_uses_transitions": False,
                "hmm_transitions_affect_reward_estimate": model.baum_welch_used,
                "uses_true_baseline": True,
                "main_comparison_uses_logged_test_activities": True,
                "decoder_uses_test_activity_labels": False,
                "top_variants_define_scope_before_split": True,
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "rq2_correlation.csv", index=False)
    return table


def run_seed_check(
    scopes: list[str] = ["top3", "top5", "learnable"],
    seeds: list[int] = ROBUSTNESS_SEEDS,
) -> pd.DataFrame:
    """Repeat the labelled regression comparison under independent noise."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope in scopes:
        base = load_problem(scope)
        for seed in seeds:
            print(f"  seed check: {scope}, noise seed {seed}")
            problem = with_noise(base, seed)
            ols = _fit_rewards(problem)
            ridge, alpha = _fit_ridge(problem)
            weighted_energy, noise_variance, event_variance, background_variance = (
                _fit_aggregate_emissions(problem)
            )
            rows.append({
                "scope": scope,
                "noise_seed": seed,
                "ols_test_rmse": _rmse(
                    problem.target, problem.X @ ols, problem.cut
                ),
                "ridge_test_rmse": _rmse(
                    problem.target, problem.X @ ridge, problem.cut
                ),
                "weighted_regression_test_rmse": _rmse(
                    problem.target, problem.X @ weighted_energy, problem.cut
                ),
                "ols_mean_cost_error": float(
                    np.mean(np.abs(ols - problem.truth))
                ),
                "ridge_mean_cost_error": float(
                    np.mean(np.abs(ridge - problem.truth))
                ),
                "weighted_regression_mean_cost_error": float(
                    np.mean(np.abs(weighted_energy - problem.truth))
                ),
                "ridge_alpha": alpha,
                "weighted_interval_noise_variance": noise_variance,
                "weighted_event_variance": event_variance,
                "weighted_background_variance": background_variance,
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "reward_estimator_seed_check.csv", index=False)
    return table


def run_transition_seed_check(
    scopes: list[str] = ["top3", "learnable"],
    seeds: list[int] = DECODER_SEEDS,
) -> pd.DataFrame:
    """Repeat the matched transition/no-transition decoder comparison.

    Five decoder repeats are used because whole-log coordinate decoding is much
    more expensive than fitting the reward coefficients.  Only measurement
    noise changes; the event log, paths, durations and chronological split stay
    fixed.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope in scopes:
        base = load_problem(scope)
        expert_count = 0 if scope == "learnable" else int(scope.removeprefix("top"))
        for seed in seeds:
            print(f"  transition check: {scope}, noise seed {seed}")
            problem = with_noise(base, seed)
            model = fit_pooled_hmm(problem, use_transitions=True)
            independent_model = fit_pooled_hmm(
                problem, use_transitions=False
            )

            independent_em_states, independent_em_counts, _ = (
                decode_without_transitions(problem, independent_model)
            )
            independent_em = _metrics(
                problem,
                independent_model,
                independent_em_states,
                independent_em_counts,
            )
            independent_states, independent_counts, _ = (
                decode_without_transitions(problem, model)
            )
            independent = _metrics(problem, model, independent_states, independent_counts)

            pooled_states, pooled_counts, _ = decode_pooled(problem, model)
            pooled = _metrics(problem, model, pooled_states, pooled_counts)

            (
                variant_states,
                variant_counts,
                coverage,
                available,
                selected,
                _,
            ) = decode_variant_first(problem, model, expert_count)
            variant = _metrics(
                problem, model, variant_states, variant_counts
            )
            ols_energy = _fit_rewards(problem)
            test_events = problem.events["bin"].to_numpy(int) >= problem.cut
            variant_lrm_rmse = _rmse(
                problem.target, variant_counts @ ols_energy, problem.cut
            )
            variant_lrm_event_mae = float(np.mean(np.abs(
                ols_energy[variant_states[test_events]]
                - problem.events.loc[test_events, "energy"].to_numpy(float)
            )))

            rows.append({
                "scope": scope,
                "noise_seed": seed,
                **_prefixed("independent", independent),
                **_prefixed("independent_em", independent_em),
                "weighted_regression_mean_cost_error": float(
                    np.mean(np.abs(model.weighted_energy - problem.truth))
                ),
                "hmm_mean_cost_error": float(
                    np.mean(np.abs(model.energy - problem.truth))
                ),
                "independent_em_mean_cost_error": float(
                    np.mean(np.abs(independent_model.energy - problem.truth))
                ),
                **_prefixed("pooled_hmm", pooled),
                **_prefixed("variant_hmm", variant),
                "variant_hmm_lrm_test_rmse": variant_lrm_rmse,
                "variant_hmm_lrm_event_energy_mae": variant_lrm_event_mae,
                "represented_test_events_percent": coverage,
                "training_variants_available": available,
                "selected_variant_experts": selected,
                "uses_true_baseline": True,
                "decoder_uses_test_activity_labels": False,
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "transition_seed_check.csv", index=False)
    return table


def run_scope_audit() -> pd.DataFrame:
    """Verify why the reported whole-system scope contains 35 activities."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    complete = _simulated_complete_events()
    # Rank is reported from whole-event counts, so it describes the process
    # rather than the hair of difference that duration sharing adds.
    _, complete_matrix = build_signal_and_matrix(complete, spread_duration=False)
    complete_activities = [
        column for column in complete_matrix.columns if column != "timestamp"
    ]
    complete_X = complete_matrix[complete_activities].to_numpy(float)
    complete_cut = event_count_split(complete_X)
    cold_start = [
        activity
        for index, activity in enumerate(complete_activities)
        if complete_X[:complete_cut, index].sum() == 0
        and complete_X[complete_cut:, index].sum() > 0
    ]

    learnable, exclusions = filter_learnable_events(complete)
    _, learnable_matrix = build_signal_and_matrix(learnable, spread_duration=False)
    learnable_activities = [
        column for column in learnable_matrix.columns if column != "timestamp"
    ]
    learnable_X = learnable_matrix[learnable_activities].to_numpy(float)
    learnable_cut = event_count_split(learnable_X)
    exclusions.to_csv(OUT_DIR / "excluded_activities.csv", index=False)

    table = pd.DataFrame([{
        "complete_events": len(complete),
        "complete_activities": len(complete_activities),
        "complete_full_rank": int(np.linalg.matrix_rank(complete_X)),
        "complete_training_rank": int(
            np.linalg.matrix_rank(complete_X[:complete_cut])
        ),
        "cold_start_activities": " | ".join(cold_start),
        "learnable_events": len(learnable),
        "learnable_event_percent": 100.0 * len(learnable) / len(complete),
        "learnable_activities": len(learnable_activities),
        "learnable_full_rank": int(np.linalg.matrix_rank(learnable_X)),
        "learnable_training_rank": int(
            np.linalg.matrix_rank(learnable_X[:learnable_cut])
        ),
    }])
    table.to_csv(OUT_DIR / "scope_audit.csv", index=False)
    return table


def main(
    scopes: list[str],
    experts: int | None,
    include_rq2: bool,
    include_seed_check: bool,
    include_transition_check: bool,
) -> None:
    """Run every experiment and save one CSV table per experiment.

    The scope audit first, then each scope in turn, then the optional
    correlation stress test and the two repeat checks that show the results are
    not a single lucky noise draw.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\nScope audit\n{'=' * 72}")
    print(run_scope_audit().to_string(index=False))
    results = []
    for scope in scopes:
        print(f"\n{'=' * 72}\n{scope}: state = activity\n{'=' * 72}")
        (
            result,
            independent_history,
            pooled_history,
            variant_history,
        ) = evaluate_scope(scope, experts)
        results.append(result)
        independent_history.to_csv(
            OUT_DIR / f"{scope}_independent_sweeps.csv", index=False
        )
        pooled_history.to_csv(OUT_DIR / f"{scope}_pooled_sweeps.csv", index=False)
        variant_history.to_csv(OUT_DIR / f"{scope}_variant_sweeps.csv", index=False)
        print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    table = pd.concat(results, ignore_index=True)
    table.to_csv(OUT_DIR / "event_state_results.csv", index=False)
    if include_rq2:
        print(f"\n{'=' * 72}\nRQ2: correlated event chains\n{'=' * 72}")
        rq2 = run_rq2()
        print(rq2.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    if include_seed_check:
        print(f"\n{'=' * 72}\nReward estimators: 30 noise seeds\n{'=' * 72}")
        seed_check = run_seed_check()
        print(
            seed_check.to_string(
                index=False, float_format=lambda value: f"{value:.5f}"
            )
        )
    if include_transition_check:
        print(f"\n{'=' * 72}\nMatched transition check\n{'=' * 72}")
        transition_check = run_transition_seed_check()
        print(
            transition_check.to_string(
                index=False, float_format=lambda value: f"{value:.5f}"
            )
        )
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event-state HMM thesis experiments")
    parser.add_argument(
        "--scope",
        default="top1,top2,top3,top5,learnable",
        help="comma-separated: top1, top2, top3, top5, learnable",
    )
    parser.add_argument(
        "--experts", type=int, default=None,
        help="variant experts per case length; use 0 for every training path",
    )
    parser.add_argument(
        "--skip-rq2", action="store_true", help="skip the correlation stress test"
    )
    parser.add_argument(
        "--seed-check",
        action="store_true",
        help="repeat OLS, ridge and weighted regression with 30 noise seeds",
    )
    parser.add_argument(
        "--transition-check",
        action="store_true",
        help="repeat matched transition/no-transition decoding with five seeds",
    )
    args = parser.parse_args()
    main(
        [item.strip() for item in args.scope.split(",")],
        args.experts,
        not args.skip_rq2,
        args.seed_check,
        args.transition_check,
    )
