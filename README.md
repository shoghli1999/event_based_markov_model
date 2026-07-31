# thesis/

Code for my event-based energy modelling experiments on the BPI Challenge 2019 log.

The idea: take process variants from the log, treat each activity as a Markov state, hide made-up energy costs inside a synthetic signal, then try to recover those costs from the signal. On top of that, `markov_reward.py` uses the learned energies + transition probabilities to get expected energy per case.

This folder is self-contained — it reads the XES file directly and writes everything locally. The rest of the repo (`signal_generator/`, `Event Log Manager/`) is older stuff from the project I built on.

Data: [BPI Challenge 2019](https://www.tf-pm.org/resources/bpi-challenge/bpi-challenge-2019) (purchase-to-pay).

---

## how to run

```bash
cd thesis
pip install pm4py pandas numpy matplotlib networkx scikit-learn scipy

python MarkovModel_clean.py      # default: top 3 variants → generated_signals_k3/
python markov_reward.py          # reads those CSVs, adds reward-process outputs
```

Other options:

```bash
python MarkovModel_clean.py --k 1    # one variant only
python MarkovModel_clean.py --k 2    # two variants

python markov_reward.py --k 2
python markov_reward.py --energy true   # use true energies instead of recovered ones
```

Comparison experiments (see the section further down):

```bash
python compare_optimizers.py       # coordinate descent vs gradient descent
python regression_comparison.py    # linear regression baseline, ridge, time split
python hmm_variants.py             # experiment 1 — HMM on variants 1+2+3
python hmm_signal.py               # experiment 2 — Markov layer on the 30-min signal
```

Run `MarkovModel_clean.py` first. Everything else only reads the CSVs it writes — no XES reload.

---

## what's in here

```
thesis/
├── BPI_Challenge_2019.xes          # input log (~695 MB)
├── MarkovModel_clean.py            # main script — use this one
├── markov_reward.py                # MRP layer on top
├── compare_optimizers.py           # coordinate descent vs gradient descent
├── regression_comparison.py        # linear regression baseline vs mine, ridge, time split
├── hmm_variants.py                 # experiment 1 — per-case HMM (Baum-Welch + Viterbi)
├── hmm_signal.py                   # experiment 2 — Markov-switching regression on the signal
├── MarkovModel.py                  # old version, 1 variant
├── MarkovModel_v2.py               # old version, 2 variants
├── MarkovModel_v3.py               # old version, 3 variants
├── generated_signals_k3/           # current outputs (k=3)
├── generated_signals/              # outputs from MarkovModel.py
├── generated_signals_v2/           # outputs from MarkovModel_v2.py
└── generated_signals_v3/           # outputs from MarkovModel_v3.py
```

The three `MarkovModel*.py` files are earlier iterations I kept for reference. `MarkovModel_clean.py` is the consolidated version with `--k`. Column names differ slightly in the old outputs (`without_noise`/`with_noise` vs `clean`/`signal`).

---

## the model (short version)

- each event type = one Markov state
- energy is paid when you *enter* a state, not on the transition
- within one variant the chain is basically deterministic (prob ≈ 1)
- when you pool variants, probabilities reflect how often each path actually happens in the log

I simulate event durations myself (normal, clipped 120–240 s). The "true" cost per event is `BASE_COST + duration × 0.01`. Those base costs are made up — they're the ground truth I hide in the signal and later try to get back.

The signal is: sine baseline (daily pattern) + Gaussian noise + event costs binned to 30 minutes.

Learning is coordinate descent on the event-occurrence matrix — the guess-and-refine loop from the thesis. I compare recovery on clean vs noisy signal to check that noise averages out.

---

## scripts

### `MarkovModel_clean.py`

1. load XES
2. find top-k variants
3. build event objects with simulated durations
4. generate synthetic signal
5. cost table + 30-min event matrix
6. Markov chains (pooled + per variant) + PNG plots
7. learn per-state energy, print summary

Config is at the top of the file: `FREQ`, `SINE_AMPLITUDE`, `NOISE_STD`, `LEARN_ITERS`, `BASE_COSTS`, etc.

### `markov_reward.py`

Reads `learning_summary.csv` for per-state energy and the markov chain CSVs for transition probs. Computes expected visits via the fundamental matrix `(I − Q)⁻¹`, multiplies by energy → expected energy per case.

The per-state energy values come from the signal (same as the baseline). The probabilities come from counting transitions. The pooled chain should match the case-weighted average of the per-variant numbers — that's the sanity check (gap should be < 1e-3).

Also writes reward-chain plots where nodes show E (energy) and n (expected visits).

---

## comparison experiments

These four never load the XES: `compare_optimizers.py` and `regression_comparison.py`
read only the CSVs in `generated_signals_k{k}/` and run in seconds; `hmm_variants.py`
and `hmm_signal.py` simulate or read the same CSVs and take a few minutes. Run
`MarkovModel_clean.py` first.

### `compare_optimizers.py`

Coordinate descent vs gradient descent vs the exact least-squares solution, on
the same X and y.

Result: all three land on the same answer. Coordinate descent needs ~10 passes,
gradient descent ~1,000 (5,000 on the noisy signal), because XᵀX has condition
number ≈ 109 — the event types differ a lot in how often they occur, which makes
gradient descent zig-zag. Rescaling the columns brings gradient descent down to
~20 steps but adds a learning rate to tune.

The residual error of 5.7e-4 is identical for every method including the exact
solution, so it comes from binning and noise, not from the optimiser.

### `regression_comparison.py`

Reproduces the linear-regression baseline from `Event Log Manager/event_cost_eval.py`
(my supervisor's implementation) — `LinearRegression(fit_intercept=False,
positive=True)` on a design matrix with a manually added ones-column — and
compares it against coordinate descent, plain OLS and ridge, under a
chronological 70/30 train/test split.

Two setups are evaluated:

- **oracle baseline** — the true sine is subtracted first, as `MarkovModel_clean.py`
  does. Convenient, but the true baseline is not knowable in practice.
- **constant baseline** — the raw signal is used and the background load is
  estimated as a single intercept, which is what the baseline does.
  This setup is *not* a realistic baseline model and is deliberately not called
  one: a constant cannot follow a daily sine, so the wave stays in the residual.
  Modelling that background load is what the Markov layer in `hmm_signal.py` is
  for — the regimes take the place of the single constant.

Results:

- The baseline's non-negative fit, plain OLS and coordinate descent agree to
  four decimals. `positive=True` never binds, because the unconstrained solution
  is already positive everywhere (energies run 55–101).
- Ridge does not help. Error grows from 8e-4 at α=0.01 to 2e-2 at α=1000. With
  18,304 rows and 5 unknowns there is no variance problem for it to fix. The
  penalty is applied through `P = diag(0, 1, 1, …)` so the intercept is genuinely
  exempt; `Ridge(fit_intercept=False)` on a matrix with a manual ones-column would
  shrink the background load too, which an earlier version of this file did.
- Going from the oracle to the constant baseline raises the worst energy error
  from 8e-4 to 2.6e-2 and test RMSE from 0.70 to 1.57. The estimated intercept is
  5.30 against a true baseline mean of 5.0. The extra error is the daily sine: a
  residual sine of amplitude 2 contributes √(1.41²+0.5²) ≈ 1.50, essentially the
  observed 1.57.

### `hmm_variants.py`

A real Hidden Markov Model, Baum-Welch and Viterbi written from scratch (no
`hmmlearn`). Hidden states are the activities; only per-step energy readings are
observed. Four methods are scored on identical data: the linear regression
baseline (labels known), a step-number baseline, the HMM, and a Gaussian mixture
(no transitions). The mixture is the control that isolates what the Markov
structure contributes.

```bash
python hmm_variants.py                    # default: variants 1+2+3
python hmm_variants.py --variants 1,2     # equal-length subset
python hmm_variants.py --variants 1       # fixed order, HMM cannot show its value
```

Variants of different lengths are supported. Variant 3 has 2 steps against 5 for
variants 1 and 2, so traces are right-padded to the longest one and carried with
a mask. Past the end of a short trace the forward recursion freezes alpha and
sets its scaling factor to 1, so padded steps contribute log 1 = 0 and the chain
cannot keep walking through the transition matrix; a transition is counted only
when both of its endpoints are real; and Viterbi holds its score with an identity
backpointer, so the traceback carries the last real state backwards. Padded slots
are excluded from every mean, variance, transition count and accuracy score.

The number of hidden states is **not** set to the true activity count. It is
searched over `K_CANDIDATES` and chosen on a validation split by per-observation
held-out log-likelihood. That score climbs, flattens around K=7 and then stays
level out to K=20, so its plain argmax picks an arbitrary K from the flat region
(it returned 15 here, on noise alone). `select_k` therefore takes the smallest K
within `SELECT_TOL` of the best — the point where the curve levels off.

State-to-activity mapping is built from the **training decoding only**
(`map_states_from_train`). An earlier version matched learned means against
activity means computed over the whole data set, which leaked test information
into the evaluation.

Results with the current `BASE_COSTS`, on variants 1+2+3 (the default):

| noise | chosen K | HMM | step number only | mixture |
|-------|----------|-----|------------------|---------|
| 0.5   | 7 | 100.0% | 83.2% | 83.9% |
| 1     | 7 | 100.0% | 83.2% | 64.1% |
| 2     | 7 | 100.0% | 83.2% | 53.1% |
| 5     | 6 |  85.3% | 83.2% | 44.1% |
| 10    | 5 |  73.4% | 83.2% | 42.6% |
| 20    | 5 |  82.1% | 83.2% | 36.1% |

So the defensible claim is narrow: an HMM helps when several variants share
activities but order them differently, and the transition information separates
states that the energies alone cannot. That advantage is not guaranteed as noise
grows — past noise 5 the step-number baseline wins.

Four honest limitations:

- **The chosen K exceeds the activity count at low noise.** With K=7 for 5
  activities, EM splits an activity that appears at different positions, and the
  many-to-one training mapping merges the pieces back. Part of the accuracy gain
  therefore comes from that mapping step, which uses training labels, rather than
  from the HMM alone. Fixing K to 5 gives 92.7% at noise 0.5 instead of 100%.
- **At high noise EM still falls into a local optimum** where the states act as
  step numbers and the transition matrix loses its branching. `_init_params`
  starts restart 0 from a Gaussian mixture, which fixes the low-noise case but
  not the high-noise one.
- **Variant 1 alone proves nothing.** The step-number baseline scores 100% by
  construction there, since the order never changes. Report variants 1+2 or
  1+2+3.

On energy recovery the HMM matches the linear regression baseline at low noise
(0.0212 vs 0.0212 at noise 0.5) and falls behind as noise grows, which is
expected — the baseline is given the labels. The HMM column is not monotone in
noise (1.93 at noise 5 against 0.58 at noise 10); that is EM landing in different
local optima, not a property of the data, and it is a reason to treat single
high-noise numbers with caution rather than to read a trend into them.

### what this experiment is, and is not

This is a **controlled per-case experiment**: one energy reading per activity
step per case. It shows that hidden states plus transition structure can recover
activities from energy alone, and how that degrades with noise.

It is **not** the aggregated-signal problem the proposal defines. The comparison
against the baseline here is also not like-for-like: the linear regression
baseline is handed the true activity label of every observation, while the HMM
must infer it. That problem is the subject of `hmm_signal.py` below, where the
comparison is fair.

### `hmm_signal.py`

Experiment 2 — the Markov layer on the aggregated 30-minute signal, which is the
problem the proposal defines:

    y_t = b_t + Σ_j X_tj w_j + ε_t

Markov-switching regression: a hidden regime `z_t` follows a Markov chain and

    y_t | z_t = k  ~  N( b_k + Σ_j X_tj w_j , σ_k² )

The energies `w` are shared across regimes — energy per event type is a physical
constant. What switches is the background level `b_k` and the noise scale `σ_k`.

**Why the comparison is fair here.** At K=1 the model is exactly the linear
regression baseline: one constant intercept, one set of energies, least squares.
So K=1 is the baseline, fitted by the same code on the same rows, and the Markov
layer is a strict generalisation of it. Both see the same y, the same X, the same
split. And because it works on intervals rather than traces, every variant in the
event matrix takes part — the equal-length restriction of `hmm_variants.py` does
not apply, so nothing is dropped.

Split is chronological: 60% train, 10% validation, 30% test. Test numbers use
one-step-ahead prediction, where ŷ_t is built from y_{<t} and X_t only.

| K | | test RMSE (1-step) | worst energy error |
|---|---|--------------------|--------------------|
| 1 | linear regression baseline | 1.5657 | 0.0257 |
| 2 | | 1.0230 | 0.0094 |
| 4 | | 0.9194 | 0.0075 |
| 6 | **chosen** | **0.8743** | **0.0068** |
| 8 | | 0.8667 | 0.0053 |
| 12 | | 0.8299 | 0.0021 |

The Markov layer cuts prediction error by 44% and energy error by a factor of
about four against the linear regression baseline, on the proposal's own data.

The six regime background levels come out as 3.39, 3.90, 4.63, 5.71, 6.53, 6.78
against a true baseline sine running 3.0…7.0. The regimes have tiled the daily
cycle — the Markov layer is recovering the background load that a single constant
intercept cannot follow.

Three things to be careful about:

- **K is chosen on validation RMSE, not validation log-likelihood.** The
  likelihood never turns over: it keeps climbing to K=48 and beyond, because
  extra regimes always buy some variance structure. Test RMSE, meanwhile, is flat
  past K≈12 and at K=48 was *worse* than at K=32. Selecting on the reported
  quantity is the defensible rule; `SELECT_TOL` on likelihood is kept only as a
  diagnostic column.
- **Gaussian HMM likelihood is unbounded.** Before `SIGMA_FLOOR_FRAC` was added a
  regime at K=48 collapsed to σ=0.016 with a background level of 15.0, far outside
  the true 3…7 range, and drove the likelihood up on nothing.
- **Test RMSE is 0.87 against a noise floor of 0.5.** Piecewise-constant regimes
  approximate a smooth daily cycle, they do not reproduce it, so a gap remains by
  construction. More regimes shrink it, which is why the candidate list is capped
  at 12: past that the background load stops being a process model and becomes a
  lookup table for the time of day.

---

## input

**`BPI_Challenge_2019.xes`** — BPI 2019 purchase-to-pay log, ~695 MB.

GitHub won't take files over 100 MB, so you'll need Git LFS or just download the log yourself and drop it in this folder before running.

---

## outputs — `generated_signals_k{k}/`

Written by `MarkovModel_clean.py`. Reward files added by `markov_reward.py`.

**CSVs from the clean model:**

- `variants_overview.csv` — which variants, how many cases, the activity sequence
- `signal.csv` — 30-min time series: baseline, noise, event_cost, clean, signal
- `cost_table.csv` — counts and costs per variant × event type
- `event_matrix_30min.csv` — how many events of each type per 30-min bin
- `markov_chain_pooled.csv` — state, next_state, count, probability (all variants mixed)
- `variant_1_markov_chain.csv` etc. — same thing but for one variant each
- `learning_summary.csv` — true vs learned energy per state (+ AVERAGE row)

**CSVs from markov_reward:**

- `markov_reward_recovery.csv` — true vs recovered energy, abs error
- `markov_reward_per_case.csv` — expected energy per case (per variant, pooled, weighted avg check)
- `markov_reward_visits.csv` — expected visits per state on the pooled chain

**CSVs from the comparison experiments:**

- `regression_comparison_metrics.csv` — per method: energy error, train/test R², test RMSE
- `regression_comparison_energies.csv` — recovered energy per state, both setups
- `hmm_variants123_comparison.csv` — HMM vs baseline vs step number vs mixture (default, variants 1+2+3)
- `hmm_variants12_comparison.csv`, `hmm_variants1_comparison.csv` — same for the 1+2 and 1-only subsets
- `hmm_signal_comparison.csv` — per K: test RMSE, energy error, regime sigmas
- `hmm_signal_k_selection.csv` — validation log-likelihood and RMSE per K

**PNGs:**

- `markov_chain_pooled.png`, `variant_*_markov_chain.png` — transition graphs
- `variant_*_reward_chain.png`, `markov_reward_chain_pooled.png` — same but with energy + visits on nodes
- `markov_reward_per_case.png` — bar chart

There's also a `markovChainPictures/` subfolder with copies of some plots I used in the thesis writeup.

---

## old output folders

Same kind of files, different naming. Kept so I don't break anything that references them.

**`generated_signals/`** (from `MarkovModel.py`, 1 variant):
`variant1_signal.csv`, `variant1_cost_table.csv`, `variant1_event_matrix_30min.csv`, `variant1_markov_chain.csv`, `variant1_markov_state_learning.csv`, `variant1_markov_state_learning_summary.csv`

**`generated_signals_v2/`** (2 variants):
shared files prefixed `two_variants_*`, per-variant chains as `variant_1_markov_chain.csv` etc., plus PNGs

**`generated_signals_v3/`** (3 variants):
same layout, prefix is `three_variants_*`

---

## deps

- pm4py (XES loading)
- pandas, numpy
- matplotlib, networkx (plots)
- scikit-learn, scipy (comparison experiments only — `LinearRegression`, `Ridge`,
  `GaussianMixture`, `linear_sum_assignment`)

No requirements.txt in the repo — just pip install what you need.

Seeds are fixed (`SIGNAL_SEED = 123` etc.) so re-running on the same log gives the same numbers.
