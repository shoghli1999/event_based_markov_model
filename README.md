# Event-based energy modelling with variant-first hidden Markov models

Code for my master's thesis at the University of Passau, submitted on 23 September
2026: *Event-Based Energy Modelling with Variant-First Hidden Markov Models*. It was
written at the Chair of Distributed Information Systems, supervised by Manuel Lehner
and examined by Prof. Dr. Harald Kosch and Prof. Dr. Michael Granitzer. It builds on
the event-based energy modelling framework of Lehner (Athens Journal of Sciences
12(4), 2025, doi:10.30958/ajs.12-4-4).

The question behind it: a facility meter records one energy total every 30 minutes,
and an event log records which process activities happened and when. Can we recover
how much energy each activity used, and attribute energy back to single events?

The central modelling decision is that one hidden state is one process activity.
The number of states is therefore fixed by the log, the transition matrix says
which activity follows which, and the emissions describe the energy and timing of
each activity.

## Summary

The result tables are in `results_event_state/`.

- Over 30 noise draws, the weighted estimator lowered the median activity-cost
  error of OLS by 34.55% on the top three variants, 55.84% on the top five and
  16.92% on the whole system.
- The variant-first decoder named at least 99.92% of hidden activities correctly
  on the four frequent-variant scopes. On the whole system it was the best of the
  four methods, with an energy error per event 26.5% below a matched control.
- The Markov reward layer predicted the energy of a complete case within 0.118%
  for top-three cases that had time to finish.

## Data

The event log is the BPI Challenge 2019 purchase-order log, published on
[4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853).
It is not included here because of its size. From it the code uses case
identifiers, activity names and timestamps. After removing 320 events
with timestamps outside 2018 and 2019, it holds 1,595,603 events in 251,734
cases and 42 activities.

The log contains no energy, so the energy signal is generated and its true costs
are known. It follows my supervisor's generator:

- each activity has a fixed base cost;
- each event gets a duration from a normal distribution with mean 180 seconds
  and standard deviation 20, clipped to 120-240 seconds;
- an event's energy is its base cost plus its duration times 0.01;
- an event that crosses an interval boundary has its energy and its activity
  count shared between the two intervals, in proportion to the time spent in
  each;
- the building background is a daily sine shape peaking at 14:00, multiplied by
  a fresh random number between 80 and 90 in every interval;
- the meter reports one total every 30 minutes.

The random seeds are fixed: 42 for the durations and 123 for the background
noise.

Before fitting, only the predictable part of the background (the daily shape
times 85) is removed. The random part stays in the signal as noise.

## Method in short

Cost estimation. Feasible generalized least squares on the interval totals. The
noise variance of an interval is modelled as

```text
variance = a + b * (events in the interval) + c * (background shape) ** 2
```

and the three terms are learned from the training residuals by non-negative
least squares. Ordinary least squares and ridge regression are the baselines.

Decoding hidden activities. After a chronological cut at about 70 percent of the
events, the activity labels of the held-out events are hidden. Three decoders
name them: a no-transition control, a pooled Viterbi decoder, and a
variant-first decoder that uses the complete paths seen in training and falls
back to the pooled model when no path fits. A position-only rule serves as a
baseline.

Reward layer. An absorbing Markov chain with an END state gives the expected
energy of one complete case, split by activity.

Scopes. Cases on the 1, 2, 3 and 5 most frequent variants, and the whole system.
The whole system keeps 35 of the 42 activities, covering 99.661 percent of the
events. Five SRM activities occur only together, in two groups at the same
timestamps, and two activities never appear before the cut, so these seven
costs cannot be learned.

## Main results

Activity cost error against the true costs, median over 30 noise draws:

| scope | OLS | weighted estimator | lower by | draws won |
|---|---:|---:|---:|---:|
| top three | 0.002710 | 0.001774 | 34.55% | 29 of 30 |
| top five | 0.035483 | 0.015670 | 55.84% | 30 of 30 |
| whole system | 0.541584 | 0.449926 | 16.92% | 24 of 30 |

Energy error per hidden event, main run:

| scope | held-out events on paths seen in training | no-transition control | variant-first | lower by |
|---|---:|---:|---:|---:|
| top one | 100.0% | 1.2537 | 0.1586 | 87.4% |
| top two | 100.0% | 1.5877 | 0.1681 | 89.4% |
| top three | 100.0% | 1.5772 | 0.1675 | 89.4% |
| top five | 100.0% | 6.6963 | 0.1977 | 97.0% |
| whole system | 58.9% | 25.9710 | 19.0767 | 26.5% |

On the whole system the variant-first decoder names 48.67% of held-out
activities correctly, against 43.20% for the position-only baseline, 28.93% for
the control and 22.06% for the pooled decoder.

Other findings:

- With known activities, rebuilding the meter is a tie. On the whole system the
  weighted estimator reaches an RMSE of 4.0096, a model given the true costs
  4.0164, and the background noise alone sets a floor of 3.7540.
- The variance model finds the background term within two percent of its true
  value (8.3333) on top two, top three and top five.
- For top-three cases that began at least a full observation margin before the
  log ends, the reward layer predicts the mean case energy within 0.118%. For
  all cases starting after the cut it predicts too much, by 5.39% on top three
  and 18.42% on the whole system, which is consistent with cases still running
  when the log ends.
- Keeping the five SRM activities as two merged groups recovers the two group
  costs closely, but the median cost error of the other 35 activities rises
  from 0.449926 to 0.519489, and it is worse in 24 of 30 draws.
- Splitting an event's energy while counting it whole in its starting interval
  leaves 7.95% of the event energy on top three in 434 intervals with no counted
  activity. This is why counts and energy are split the same way.
- Baum-Welch is implemented but switched off, because it raises the cost error
  on all five scopes. It can be switched on with `BAUM_WELCH_DEFAULT` or the
  `baum_welch` argument of `fit_pooled_hmm`.
- Factorial composition, with separate costs per variant, has a higher cost
  error than shared costs on all three scopes where it runs.

## Limitations

- Energy, durations, background and noise are generated; only the process data
  is real.
- The daily background shape is removed as known, and the generated durations
  decide how an event is split between intervals. A real log has no durations,
  so a real analysis would have to estimate both.
- Decoding is retrospective: case membership, event times and case lengths are
  known. Test activity labels are used only for scoring.
- On the whole system, 41.10% of held-out events belong to cases whose path was
  never seen in training, and the meter rebuilt from variant-first paths is about
  five times further off than with the pooled decoder (RMSE 304.85 against
  61.16).
- On the whole system the cost comparison depends on the noise draw, so it is
  reported as a median over 30 draws.

## Files

| file | purpose |
|---|---|
| `data_pipeline.py` | loads the log, generates the energy, builds the meter series and count matrix |
| `event_state_hmm.py` | the model, the decoders and the experiments for the four research questions |
| `plot_results.py` | the overview, stress-test and composition figures |
| `final_model/reward_layer.py` | expected energy of one complete case |
| `final_model/per_activity_check.py` | the case energy split activity by activity |
| `final_model/scope_progression.py` | why the whole system keeps 35 of 42 activities |
| `final_model/merged_scope_check.py` | merging the co-occurring SRM activities instead of removing them |
| `final_model/baum_welch_check.py` | the Baum-Welch update switched on and off |
| `final_model/factorial_check.py` | factorial composition, one set of costs per variant |
| `final_model/interval_convention_check.py` | three ways of placing an event that crosses an interval boundary |
| `final_model/thesis_facts.py` | descriptive numbers quoted in the thesis that no result table holds |
| `final_model/plot_reward.py`, `plot_per_activity.py` | figures for the reward layer and the per-activity check |
| `final_model/plot_pipeline.py` | the pipeline diagram |
| `final_model/plot_state_example.py` | selected states and transitions of the whole-system model |
| `final_model/plot_end_example.py` | the same states with the END state of the reward layer |
| `final_model/plot_split_example.py` | the chronological split and the three kinds of case |
| `final_model/plot_decoder_example.py` | the four decoding methods sketched on one case |
| `final_model/plot_signal_example.py` | two days of the generated signal and its background |
| `results_event_state/` | all result tables and figures |
| `images/` | the figures used in the thesis |

Every result figure is drawn from a saved table, so a figure and the numbers in
the thesis come from the same file.

## Running the experiment

Put `BPI_Challenge_2019.csv` next to the scripts, or set the environment variable
`BPI2019_CSV` to its path. The code reads the columns `case concept:name`,
`event concept:name` and `event time:timestamp` (cp1252 encoding). Then run, in this order:

```bash
python3 -m pip install -r requirements.txt
python3 event_state_hmm.py \
  --scope top1,top2,top3,top5,learnable \
  --seed-check \
  --transition-check
python3 final_model/scope_progression.py
python3 final_model/merged_scope_check.py
python3 final_model/baum_welch_check.py
python3 final_model/reward_layer.py && python3 final_model/plot_reward.py
python3 final_model/per_activity_check.py && python3 final_model/plot_per_activity.py
python3 final_model/factorial_check.py
python3 final_model/interval_convention_check.py
python3 final_model/thesis_facts.py
python3 plot_results.py
python3 final_model/plot_pipeline.py
python3 final_model/plot_state_example.py
python3 final_model/plot_end_example.py
python3 final_model/plot_split_example.py
python3 final_model/plot_decoder_example.py
python3 final_model/plot_signal_example.py
```

`thesis_facts.py` has to run before `plot_results.py`, because the overview
figure reads the floor of the attribution error from its output.

## Earlier versions

The repository started with a Markov model whose hidden states were unknown
energy levels, so the number of states had to be chosen. That approach was
replaced by one state per activity. The earlier code is kept in the Git history
under the tag `proposal-model` (commit `fd39855`).

## Citation

Shoghli, S. (2026). *Event-Based Energy Modelling with Variant-First Hidden Markov
Models*. Master's thesis, University of Passau.
