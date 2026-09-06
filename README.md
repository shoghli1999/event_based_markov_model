# Event-based energy modelling with variant-first hidden Markov models

Master's thesis experiment. One idea runs through the whole repository:

> **One hidden state is one process activity.**

The number of states is therefore not a free choice. It is the number of
activities. The transition matrix describes which activity follows which, and
the Gaussian emission describes how much energy an activity uses. The daily
building background belongs to the signal generator, not to the states.

## What is real and what is generated

From the BPI Challenge 2019 purchase-order log: case identifiers, activity
names, event timestamps, event order and process variants. 1,595,603 events
after removing 320 timestamps outside 2018 and 2019, across 42 activities.

Generated, following my supervisor's generator: activity energy values, event
durations, the daily background and its noise.

So this tests a method on real process structure against a known synthetic
energy truth. It does not use measured facility energy and does not claim real
energy savings.

## How the signal is built

Each activity has a fixed base cost. Each event is given a duration drawn from a
normal distribution with mean 180 seconds and standard deviation 20, clipped
between 120 and 240. An event's energy is its base cost plus its duration times
0.01. The building background is a sine peaking at 14:00, multiplied by a fresh
random number between 80 and 90 in every interval. The meter reports one total
every 30 minutes.

An event lasting about three minutes can cross a 30-minute boundary. Its energy
**and** its activity count are then shared between the two intervals in
proportion to the time spent in each, following my supervisor's generator. Both
sides move together, so the measured signal and the count matrix always describe
the same thing. `final_model/interval_convention_check.py` measures what the
alternative would have cost.

Before fitting, only the predictable part of the background is removed. The
random multiplier stays in the signal as real noise.

## How an event is placed on the timeline, and why

An event lasts about three minutes; the meter reports every thirty. So an event
can start just before one interval ends and finish in the next. There are three
ways to handle that, and all three are measured in
`final_model/interval_convention_check.py`.

| placement | energy | activity count |
|---|---|---|
| whole event | charged to the starting interval | charged to the starting interval |
| **both shared (this thesis)** | **split between the two intervals** | **split the same way** |
| energy only | split between the two intervals | left whole in the starting interval |

My supervisor's generator splits an event's cost by duration, and this thesis
follows it. His evaluation code, separately, describes the event matrix as counts
of events per interval, which is the third row. Measured on the same data, that
third combination does not work:

| scope | placement | OLS | this model | intervals with energy but no count |
|---|---|---:|---:|---:|
| top three | whole event | 0.000551 | 0.000700 | 10,561 |
| top three | both shared | 0.001298 | 0.000969 | 10,127 |
| top three | energy only | 7.923744 | 7.923744 | 10,561 |
| whole system | whole event | 0.388156 | 0.351772 | 22,652 |
| whole system | both shared | 0.492592 | 0.573062 | 22,377 |
| whole system | energy only | 98.902825 | 96.347041 | 22,652 |

Splitting the energy while counting whole events leaves energy in thousands of
intervals whose count row is empty. No estimator can explain energy where nothing
is recorded as happening, and both methods degrade by two to four orders of
magnitude, equally. So the two sides of the regression must be built the same
way. Splitting both is this thesis's decision, and the table is the reason.

Note also that the first row is the easiest of the three for everyone. Easier is
not better here: with every interval equally clean there is nothing for variance
weighting to do, which is precisely the contribution being tested.

## How activity costs are estimated

Feasible generalized least squares on the interval totals. An interval can be
noisy for two reasons, and the variance model has one term for each:

```text
variance = a + b * (events in the interval) + c * (background shape) ** 2
```

The three terms are learned from the training residuals by non-negative least
squares. Nothing is read from the generator. The model recovers them well:

| scope | fixed term | per event | background |
|---|---:|---:|---:|
| top1 | 0.08813 | 0.05083 | 7.98211 |
| top2 | 0.45174 | 0.02659 | 8.27408 |
| top3 | 0.47511 | 0.02645 | 8.24892 |
| top5 | 0.00000 | 0.06419 | 8.16737 |
| whole system | 0.00000 | 0.08846 | 7.12207 |

The true values are 0 for the fixed term, 0.0400 per event and 8.33333 for the
background.

## Baum–Welch is implemented and switched off

The log names the activity of every training event, so nothing about the hidden
states is unknown while the rewards are learned, and the per-event observation
has to be invented from each interval's leftover. Measured on all five scopes it
loses accuracy on all five. `final_model/baum_welch_check.py` produces the
table. It can be switched back on with `BAUM_WELCH_DEFAULT` or the `baum_welch`
argument of `fit_pooled_hmm`.

## Scope: 35 of 42 activities

Five SRM activities always occur together in two groups, so their individual
costs cannot be separated. Two more never appear before the chronological cut.
Removing all seven costs 0.339% of the events and leaves a full-rank problem.
Rank is measured from whole-event counts, because sharing an event across
intervals separates always-together columns by a hair of random duration and
would claim more than the data supports.

## Results

Activity-cost error, median over 30 independent noise draws. This is the number
to quote, because on the whole system the difference between methods is smaller
than the swing between draws.

| scope | OLS | this model | reduction | wins |
|---|---:|---:|---:|---:|
| top3 | 0.002710 | **0.001774** | 34.55% | 29/30 |
| top5 | 0.035483 | **0.015670** | 55.84% | 30/30 |
| whole system | 0.541584 | **0.449926** | 16.92% | 24/30 |

Hidden-activity attribution. Test activity names are removed and the decoder has
only the meter and the process order it learned.

| scope | paths recognised | no transitions | variant-first | reduction |
|---|---:|---:|---:|---:|
| top1 | 100.0% | 1.2537 | **0.1586** | 87.4% |
| top2 | 100.0% | 1.5877 | **0.1681** | 89.4% |
| top3 | 100.0% | 1.5772 | **0.1675** | 89.4% |
| top5 | 100.0% | 6.6963 | **0.1977** | 97.0% |
| whole system | 58.9% | 25.9710 | **19.0767** | 26.5% |

Reconstruction with known activities is a **tie**, about 4.01 for every method on
the whole system. That is not a shortfall. The measured noise floor is 3.7540 and
a perfect model given the true costs scores 4.0164, so every method is already at
the ceiling. The remaining error is random background that nobody can predict.

## Limitations, stated openly

- Reconstruction is a tie **when the activities are known**, not a win. When
  activities are hidden on the whole log, reconstruction is worse, see the point
  below.
- On the whole log, variant-first decoding improves attribution by 26% but makes
  interval reconstruction about four times worse, because 41% of test cases
  follow paths never seen in training and are forced onto the nearest known one.
  This does not happen on any scope with full path coverage.
- The whole-system cost advantage is small next to seed variation, so it is
  quoted as a median over 30 seeds.
- Energy, duration, background and noise are synthetic.
- **An event's duration decides how it is split across two intervals, and those
  durations are synthetic.** The real log has timestamps but no durations, so a
  real analyst could not perform this split exactly. This is a controlled
  assumption of the same kind as removing the known background shape.
- "Whole system" means the 35 learnable activities, which is 99.661% of events,
  not the literal 42-activity log.
- Decoding is retrospective: case boundaries, event times and case lengths are
  known. Test activity labels are used only for scoring.
- The number of states was not selected by cross-validation, because a state is
  an activity and the count is fixed. The proposal's dwell-time constraint does
  not apply for the same reason.
- Variants are computed directly from the log rather than in Fluxicon Disco.

## Files

| file | purpose |
|---|---|
| `data_pipeline.py` | load the log, generate energy, build the signal and count matrix |
| `event_state_hmm.py` | the model and all four research questions |
| `plot_results.py` | the three main figures |
| `final_model/reward_layer.py` | expected energy of one complete case |
| `final_model/per_activity_check.py` | the same result opened up activity by activity |
| `final_model/scope_progression.py` | why the whole system keeps 35 of 42 activities |
| `final_model/baum_welch_check.py` | measured evidence for switching Baum–Welch off |
| `final_model/factorial_check.py` | RQ4 factorial composition |
| `final_model/interval_convention_check.py` | what the interval convention costs |
| `final_model/plot_reward.py`, `plot_per_activity.py` | figures for the two checks above |
| `final_model/plot_pipeline.py` | the pipeline diagram, drawn from code |
| `results_event_state/` | every CSV table and figure |
| `images/` | the same figures, for the thesis |

## History of this repository

This repository began as an exploration and became a single, verified experiment.
The earlier files are not deleted from the project's history, only from its
current state, and the commit `fd39855` is tagged so that the earlier work stays
one click away.

What was there before: several successive versions of a Markov model built
around energy regimes rather than activities, a set of generated signal folders
for individual variants, and a number of one-off comparison scripts. That line of
work answered a different modelling question, where a hidden state was an unknown
energy level and the number of states had to be chosen. It was superseded by the
decision that one hidden state is one process activity, which removes the state
selection problem entirely and is the basis of everything here.

What is here now: one data pipeline, one model, one plotting script, and nine
small scripts under `final_model/`, six of which answer a single question a
reader might raise and three of which draw figures. Every result table and every figure in
`results_event_state/` is produced by those scripts, and every figure is drawn
only from a saved table, so a figure can never disagree with a number.

Not published here: the working notes written while the experiment was being
built, and the LaTeX drafting folder. They are kept locally because they are
about writing the thesis rather than about running the code.

## Run everything

```bash
python3 -m pip install -r requirements.txt
python3 event_state_hmm.py \
  --scope top1,top2,top3,top5,learnable \
  --seed-check \
  --transition-check
python3 final_model/scope_progression.py
python3 final_model/baum_welch_check.py
python3 final_model/reward_layer.py && python3 final_model/plot_reward.py
python3 final_model/per_activity_check.py && python3 final_model/plot_per_activity.py
python3 final_model/factorial_check.py
python3 final_model/interval_convention_check.py
python3 plot_results.py
python3 final_model/plot_pipeline.py
```

Put `BPI_Challenge_2019.csv` beside the scripts, or set the `BPI2019_CSV`
environment variable to its full path.
