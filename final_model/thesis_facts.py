"""
thesis_facts.py
───────────────
Every descriptive number quoted in the thesis that no result table holds.

The result tables in results_event_state/ carry the comparisons: cost errors,
decoder scores, the reward layer and the checks. The text also quotes numbers
that describe the log, the generated signal and the worked examples, such as
how many events cross an interval boundary, how busy the busiest held-out
interval is, or what a model given the true costs would score. This script
recomputes each of them from the same code and saves them in one table, so that
every number in the thesis can be traced to a file and a row.

Nothing here changes a result. It only reads the log, the generator and the
fitted models.

Output
──────
    results_event_state/thesis_facts.csv
        section   the thesis section the number belongs to
        fact      a short name for the number
        value     the number itself, unrounded
        note      what it means

Usage
─────
    python final_model/thesis_facts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "final_model"))

import data_pipeline as pipeline  # noqa: E402
import event_state_hmm as model  # noqa: E402
import reward_layer  # noqa: E402
from factorial_check import variant_blocks  # noqa: E402

TABLE = model.OUT_DIR / "thesis_facts.csv"
STEP = pd.Timedelta(model.FREQ)
EXAMPLE_CASE = "4507000227_00010"
EXAMPLE_INTERVAL = pd.Timestamp("2018-04-30 09:00")
# Variance of a multiplier drawn uniformly over BACKGROUND_RANGE.
TRUE_BACKGROUND = (pipeline.BACKGROUND_RANGE[1] - pipeline.BACKGROUND_RANGE[0]) ** 2 / 12.0
ROWS: list[dict] = []


def add(section: str, fact: str, value, note: str = "") -> None:
    """Record one number under the section of the thesis that quotes it."""
    ROWS.append({"section": section, "fact": fact, "value": value, "note": note})


def timeline_start(problem) -> pd.Timestamp:
    """Timestamp of interval 0 of a scope's timeline."""
    return problem.events["ts"].dt.floor(model.FREQ).min()


def dataset(events: pd.DataFrame) -> None:
    """Size of the raw and cleaned log, the example case, variants and flows."""
    section = "Dataset"
    raw_path = pipeline._csv_path()
    columns = pd.read_csv(raw_path, nrows=0, encoding="cp1252").columns
    add(section, "raw_columns", len(columns), "attributes in the CSV export of the log")
    category = next(column for column in columns if "Item Category" in column)
    raw = pd.read_csv(
        raw_path,
        usecols=["case concept:name", "event time:timestamp", category],
        dtype=str,
        encoding="cp1252",
    )
    stamps = pd.to_datetime(raw["event time:timestamp"], dayfirst=True, errors="coerce")
    invalid = ~stamps.dt.year.between(2018, 2019)
    add(section, "dropped_events", int(invalid.sum()), "timestamps outside 2018 and 2019")
    for year, count in stamps[invalid].dt.year.value_counts().sort_index().items():
        add(section, f"dropped_events_year_{int(year)}", int(count))

    add(section, "events", len(events))
    add(section, "cases", events["case"].nunique())
    add(section, "activities", events["activity"].nunique())

    flows = raw.loc[~invalid].groupby("case concept:name")[category].first().value_counts()
    for name, count in flows.items():
        add(section, f"flow_cases:{name}", int(count), "cases per purchasing flow")
        add(section, f"flow_percent:{name}", 100.0 * count / flows.sum())

    paths = events.groupby("case", sort=False)["activity"].apply(tuple).value_counts()
    add(section, "distinct_variants", len(paths))
    add(section, "most_frequent_variant_cases", int(paths.iloc[0]))
    add(section, "top_five_variants_percent", 100.0 * paths.iloc[:5].sum() / paths.sum())

    case = events[events["case"] == EXAMPLE_CASE].sort_values("ts")
    for position, row in enumerate(case.itertuples()):
        add("Synthetic Energy Signal", f"example_event_{position}:{row.activity}:base_cost",
            pipeline.BASE_COSTS[row.activity])
        add("Synthetic Energy Signal", f"example_event_{position}:{row.activity}:duration_seconds",
            float(row.duration))
        add("Synthetic Energy Signal", f"example_event_{position}:{row.activity}:energy",
            float(row.energy))
    gap = (case["ts"].iloc[1] - case["ts"].iloc[0]).total_seconds() / 60.0
    add("The Event State Hidden Markov Model", "example_gap_minutes", gap,
        "minutes from the first to the second event of the example case")
    add("The Event State Hidden Markov Model", "example_log_gap", float(np.log1p(gap)))


def signal(events: pd.DataFrame) -> None:
    """Range of base costs, noise spread over the day, and boundary crossing."""
    section = "Synthetic Energy Signal"
    costs = pipeline.base_cost_table(sorted(events["activity"].unique()))
    add(section, "base_cost_min", min(costs.values()))
    add(section, "base_cost_max", max(costs.values()))
    hours = pd.DatetimeIndex([pd.Timestamp("2018-01-01 02:00"), pd.Timestamp("2018-01-01 14:00")])
    shape = pipeline.sine_baseline(hours)
    spread = (pipeline.BACKGROUND_RANGE[1] - pipeline.BACKGROUND_RANGE[0]) / np.sqrt(12.0)
    add(section, "noise_std_at_02", float(shape[0] * spread), "standard deviation of the random background")
    add(section, "noise_std_at_14", float(shape[1] * spread))

    section = "Interval Aggregation"
    interval = events["ts"].dt.floor(model.FREQ)
    share = pipeline._duration_share(events, interval)
    energy = events["energy"].to_numpy(float)
    add(section, "crossing_events_percent", 100.0 * float((share > 0).mean()),
        "events that cross an interval boundary, whole log")
    add(section, "crossing_energy_percent", 100.0 * float((energy * share).sum() / energy.sum()),
        "event energy carried into the following interval, whole log")


def scopes() -> dict:
    """Size of every scope, and the problems reused by the later sections."""
    section = "Process Variants and Scopes"
    problems = {}
    total_cases = model._simulated_complete_events()["case"].nunique()
    for scope in ["top1", "top2", "top3", "top5", "learnable"]:
        problem = model.load_problem(scope)
        problems[scope] = problem
        cases = problem.events["case"].nunique()
        add(section, f"{scope}:cases", cases)
        add(section, f"{scope}:cases_percent", 100.0 * cases / total_cases)
        add(section, f"{scope}:events", len(problem.events))
        add(section, f"{scope}:cut_interval", problem.cut)
        add(section, f"{scope}:intervals", len(problem.target))
    return problems


def identifiability(events: pd.DataFrame, learnable) -> None:
    """The co-occurring groups and the two activities that start after the cut."""
    section = "Identifiability and the Reported Scope"
    groups = {
        "srm_group_a": ["SRM: Awaiting Approval", "SRM: Complete", "SRM: Document Completed"],
        "srm_group_b": ["SRM: Held", "SRM: Incomplete"],
    }
    for name, members in groups.items():
        together = events[events["activity"].isin(members)].groupby(["case", "ts"])["activity"].nunique()
        add(section, f"{name}:case_timestamp_pairs", len(together))
        add(section, f"{name}:pairs_with_all_members", int((together == len(members)).sum()))
    cut_time = timeline_start(learnable) + learnable.cut * STEP
    add(section, "whole_system_cut_time", str(cut_time))
    for activity in ["Release Purchase Requisition", "SRM: Transfer Failed (E.Sys.)"]:
        stamps = events.loc[events["activity"] == activity, "ts"]
        add(section, f"{activity}:events", len(stamps))
        add(section, f"{activity}:first", str(stamps.min()))
        add(section, f"{activity}:last", str(stamps.max()))


def estimation(top3) -> None:
    """The worked interval, its Baum-Welch share, and the variance example."""
    section = "Estimating Activity Costs"
    index = int((EXAMPLE_INTERVAL - timeline_start(top3)) / STEP)
    for position, activity in enumerate(top3.activities):
        if top3.X[index, position] > 0:
            add(section, f"example_interval_count:{activity}", float(top3.X[index, position]))
    add(section, "example_interval_meter", float(top3.target[index]), "expected background removed")
    add(section, "example_interval_true_cost_total", float(top3.X[index] @ top3.truth))
    add(section, "example_interval_in_training", bool(index < top3.cut))
    add(section, "top3_training_intervals", top3.cut)

    fitted = model.fit_pooled_hmm(top3)
    high = fitted.interval_noise_variance + fitted.event_variance * 80 + fitted.background_variance * 2.1 ** 2
    low = fitted.interval_noise_variance + fitted.event_variance * 2 + fitted.background_variance * 0.1 ** 2
    add(section, "variance_ratio_14h_80_events_to_02h_2_events", float(high / low))

    section = "The Baum-Welch Update and Why It Is Not Used"
    residual = top3.target[index] - top3.X[index] @ fitted.weighted_energy
    share = residual / max(top3.X[index].sum(), 1.0)
    add(section, "example_interval_residual_share", float(share))
    goods = top3.activities.index("Record Goods Receipt")
    starting = top3.events[(top3.events["bin"] == index) & (top3.events["state"] == goods)]
    add(section, "example_constructed_goods_receipt_value", float(fitted.weighted_energy[goods] + share))
    for position, value in enumerate(starting["energy"]):
        add(section, f"example_goods_receipt_true_energy_{position}", float(value))

    section = "Composition of Variant Models"
    ols = model._fit_rewards(top3)
    add(section, "top3_goods_receipt_cost_ols", float(ols[goods]))
    add(section, "top3_goods_receipt_cost_weighted", float(fitted.weighted_energy[goods]))


def split(events: pd.DataFrame, learnable) -> None:
    """The whole-system cut, a positional cut for contrast, and the case groups."""
    section = "Train and Test Split"
    bins = learnable.events["bin"].to_numpy(int)
    add(section, "training_events", int((bins < learnable.cut).sum()))
    add(section, "test_events", int((bins >= learnable.cut).sum()))
    positional = int(0.7 * len(learnable.target))
    add(section, "test_events_after_positional_cut", int((bins >= positional).sum()),
        "a cut after seventy percent of the intervals instead of the events")
    add(section, "last_event_time", str(events["ts"].max()))
    first = learnable.events.groupby("case")["bin"].min()
    last = learnable.events.groupby("case")["bin"].max()
    add(section, "cases_completed_before_cut", int((last < learnable.cut).sum()))
    add(section, "cases_spanning_cut", int(((first < learnable.cut) & (last >= learnable.cut)).sum()))
    add(section, "cases_beginning_after_cut", int((first >= learnable.cut).sum()))


def reward(events: pd.DataFrame, learnable) -> None:
    """The example path, the end of the recording and the observation margin."""
    section = "The Markov Reward Layer"
    fitted = model.fit_pooled_hmm(learnable)
    variants = reward_layer.variant_visits(learnable).sort_values("cases", ascending=False)
    top = variants.iloc[0]
    add(section, "most_frequent_training_path_cases", int(top["cases"]))
    add(section, "most_frequent_training_path_expected_energy", float(top["visits"] @ fitted.energy))
    end = reward_layer.recording_end_timestamp()
    add(section, "recording_end", str(end))
    add(section, "events_after_recording_end", int((events["ts"] > end).sum()))
    stamps = learnable.events.groupby("case", sort=False)["ts"]
    training = list(model._complete_training_cases(learnable.events, learnable.cut))
    margin = (stamps.max()[training] - stamps.min()[training]).quantile(reward_layer.SETTLED_QUANTILE)
    add(section, "observation_margin_days", margin / pd.Timedelta(days=1))
    cut_time = timeline_start(learnable) + learnable.cut * STEP
    add(section, "cut_to_recording_end_days", (end - cut_time) / pd.Timedelta(days=1))


def compression(top3) -> None:
    """How long top-three cases last, and the example case at each level."""
    section = "The Correlated Chain Stress Test"
    events = top3.events
    stamps = events.groupby("case")["ts"]
    duration = stamps.max() - stamps.min()
    add(section, "top3_median_case_days", duration.median() / pd.Timedelta(days=1))
    for span in model.RQ2_SPANS:
        add(section, f"top3_cases_shortened_percent:{span:g}_minutes",
            100.0 * float((duration > pd.Timedelta(minutes=span)).mean()))
    paths = events.groupby("case", sort=False)["state"].apply(tuple)
    common = paths[paths == paths.value_counts().index[0]].index
    case = (duration[common] - duration.median()).abs().idxmin()
    add(section, "example_case", case, "most frequent path, duration closest to the median")
    rows = events[events["case"] == case].sort_values("ts")
    start = rows["ts"].iloc[0].normalize()
    for span in [None] + list(model.RQ2_SPANS):
        moved = rows["ts"] if span is None else model._compress_case_times(events, span)[rows.index]
        label = "original" if span is None else f"{span:g}_minutes"
        for activity, stamp in zip(rows["state"], moved):
            offset = (stamp.floor("min").normalize() - start).days
            add(section, f"example:{label}:{top3.activities[activity]}",
                f"day {offset}, {stamp.floor('min'):%H:%M}")


def factorial(problems: dict) -> None:
    """Per-variant estimates on top three and the rank without empty columns."""
    section = "Composition of Variant Models"
    for scope in ["top2", "top3", "top5"]:
        problem = problems[scope]
        blocks, variants = variant_blocks(problem)
        occupied = blocks.sum(axis=0) > 0
        add(section, f"{scope}:empty_columns", int((~occupied).sum()))
        add(section, f"{scope}:rank_without_empty_columns",
            int(np.linalg.matrix_rank(blocks[: problem.cut][:, occupied])))
        add(section, f"{scope}:non_empty_columns", int(occupied.sum()))
        if scope != "top3":
            continue
        estimates, _, _ = model.weighted_fit(
            blocks[: problem.cut], problem.target[: problem.cut], problem.background_shape[: problem.cut]
        )
        K = len(problem.activities)
        order = problem.activities.index("Create Purchase Order Item")
        for variant in range(variants):
            if occupied[variant * K + order]:
                add(section, f"top3_factorial_create_purchase_order_item:variant_{variant + 1}",
                    float(estimates[variant * K + order]))
        for column in np.flatnonzero(~occupied):
            add(section, f"top3_empty_column:variant_{column // K + 1}:{problem.activities[column % K]}", True,
                "this activity never occurs in this variant")


def metrics(problems: dict) -> None:
    """Burstiness of the held-out period and the floor of the attribution error."""
    section = "Evaluation Metrics"
    learnable = problems["learnable"]
    cut = learnable.cut
    starts = model.whole_event_counts(learnable)[cut:].sum(axis=1)
    meter = learnable.target[cut:]
    busiest = np.sort(starts)[::-1]
    top = int(0.01 * len(starts))
    add(section, "held_out_intervals", len(starts))
    add(section, "held_out_events", int(starts.sum()))
    add(section, "held_out_intervals_without_event_percent", 100.0 * float((starts == 0).mean()))
    add(section, "busiest_one_percent_intervals", top)
    add(section, "busiest_one_percent_event_share_percent", 100.0 * busiest[:top].sum() / starts.sum())
    add(section, "busiest_interval_events", int(busiest[0]))
    add(section, "busiest_interval_meter", float(meter[int(np.argmax(starts))]))
    for scope, problem in problems.items():
        test = problem.events["bin"].to_numpy(int) >= problem.cut
        states = problem.events["state"].to_numpy(int)
        energy = problem.events["energy"].to_numpy(float)
        add(section, f"{scope}:attribution_error_floor",
            float(np.mean(np.abs(problem.truth[states[test]] - energy[test]))),
            "every activity and every cost exactly right")
    goods = learnable.activities.index("Record Goods Receipt")
    add(section, "goods_receipt_average_true_energy", float(learnable.truth[goods]))


def reconstruction(problems: dict) -> None:
    """The noise floor, the perfect model, and where the error of top five sits."""
    section = "RQ1, Reconstruction of the Aggregate Signal"
    for scope, problem in problems.items():
        cut = problem.cut
        noise = (problem.target - problem.event_cost)[cut:]
        perfect = problem.target[cut:] - problem.X[cut:] @ problem.truth
        duration = (problem.event_cost - problem.X @ problem.truth)[cut:]
        add(section, f"{scope}:noise_floor_rmse", float(np.sqrt(np.mean(noise ** 2))))
        add(section, f"{scope}:perfect_model_rmse", float(np.sqrt(np.mean(perfect ** 2))))
        add(section, f"{scope}:perfect_model_variance", float(np.mean(perfect ** 2)))
        add(section, f"{scope}:background_variance_part", float(np.mean(noise ** 2)))
        add(section, f"{scope}:duration_variance_part", float(np.mean(duration ** 2)))

    top5 = problems["top5"]
    cut = top5.cut
    busiest = np.argsort(-top5.X[cut:].sum(axis=1))[: int(0.01 * (len(top5.target) - cut))]
    fitted = model.fit_pooled_hmm(top5)
    for name, costs in [("ols", model._fit_rewards(top5)), ("weighted", fitted.weighted_energy)]:
        residual = top5.target[cut:] - top5.X[cut:] @ costs
        add(section, f"top5:{name}_squared_error_share_in_busiest_one_percent",
            float(np.sum(residual[busiest] ** 2) / np.sum(residual ** 2)))


def scaling(learnable) -> None:
    """Whole-system held-out events split by whether their complete path was seen in training.

    The split uses the same candidate narrowing as the decoder and the same test
    of coverage as _seen_path_coverage, so a case counts as seen exactly when the
    reported coverage counts it. It reads held-out paths only to score, after both
    decoders have finished.
    """
    section = "RQ3, Scaling Across Variants"
    fitted = model.fit_pooled_hmm(learnable)
    variant_states, *_ = model.decode_variant_first(learnable, fitted, model.FULL_LOG_EXPERTS_PER_LENGTH)
    pooled_states, *_ = model.decode_pooled(learnable, fitted)

    events = learnable.events
    test = events["bin"].to_numpy(int) >= learnable.cut
    logged = events["state"].to_numpy(int)
    energy = events["energy"].to_numpy(float)
    templates, _, _ = model._training_templates(learnable, model.FULL_LOG_EXPERTS_PER_LENGTH)
    seen = np.zeros(len(events), dtype=bool)
    cases = []
    for _, case in events.groupby("case", sort=False):
        rows = case.index.to_numpy()
        positions = np.arange(len(rows))
        held_out = positions[test[rows]]
        if not len(held_out):
            continue
        candidates = templates.get(len(rows), [])
        known = positions[~test[rows]]
        if len(known):
            candidates = [
                item for item in candidates
                if np.array_equal(item[0][known], logged[rows[known]])
            ] or candidates
        covered = any(np.array_equal(path, logged[rows]) for path, _ in candidates)
        seen[rows[held_out]] = covered
        cases.append((covered, len(candidates)))
    cases = pd.DataFrame(cases, columns=["covered", "candidates"])

    add(section, "held_out_cases", len(cases))
    add(section, "held_out_cases_seen_percent", 100.0 * float(cases["covered"].mean()))
    add(section, "median_candidates_seen_cases", float(cases.loc[cases["covered"], "candidates"].median()),
        "candidate paths of the same length after narrowing by the events before the cut")
    add(section, "median_candidates_unseen_cases", float(cases.loc[~cases["covered"], "candidates"].median()))
    for label, mask in [("seen", test & seen), ("unseen", test & ~seen), ("all", test)]:
        add(section, f"{label}:held_out_events", int(mask.sum()))
        for name, states in [("pooled", pooled_states), ("variant_first", variant_states)]:
            add(section, f"{label}:{name}_attribution_error",
                float(np.mean(np.abs(fitted.energy[states[mask]] - energy[mask]))))
            add(section, f"{label}:{name}_state_accuracy_percent",
                100.0 * float(np.mean(states[mask] == logged[mask])))


def noise_sources(problems: dict) -> None:
    """True per-event variance after clipping, conditioning, and the example interval."""
    section = "RQ1, Recovery of the Two Noise Sources"
    for scope, problem in problems.items():
        duration = problem.events["duration"].to_numpy(float)
        add(section, f"{scope}:realized_per_event_variance",
            float((pipeline.DURATION_SCALE * duration).var()))
    for scope in ["top3", "learnable"]:
        problem = problems[scope]
        cut = problem.cut
        design = np.column_stack(
            [np.ones(cut), problem.X[:cut].sum(axis=1), problem.background_shape[:cut] ** 2]
        )
        eigenvalues = np.linalg.eigvalsh(design.T @ design)
        add(section, f"{scope}:variance_design_eigenvalue_ratio", float(eigenvalues.max() / eigenvalues.min()))

    seeds = pd.read_csv(model.OUT_DIR / "reward_estimator_seed_check.csv")
    whole = seeds[seeds["scope"] == "learnable"].median(numeric_only=True)
    estimated = (whole["weighted_interval_noise_variance"] + whole["weighted_event_variance"] * 80
                 + whole["weighted_background_variance"] * 2.1 ** 2)
    true = (pipeline.DURATION_SCALE * pipeline.DURATION_STD) ** 2 * 80 + TRUE_BACKGROUND * 2.1 ** 2
    add(section, "whole_system_variance_14h_80_events_median_terms", float(estimated))
    add(section, "whole_system_variance_14h_80_events_true_terms", float(true))


def main() -> None:
    """Recompute every descriptive number and save them in one table."""
    model.OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = model._simulated_complete_events().copy()
    dataset(events)
    signal(events)
    problems = scopes()
    identifiability(events, problems["learnable"])
    estimation(problems["top3"])
    split(events, problems["learnable"])
    reward(events, problems["learnable"])
    compression(problems["top3"])
    factorial(problems)
    metrics(problems)
    reconstruction(problems)
    noise_sources(problems)
    scaling(problems["learnable"])
    table = pd.DataFrame(ROWS)
    table.to_csv(TABLE, index=False)
    print(table.to_string(index=False))
    print(f"\nsaved -> {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
