# thesis/

Code for my event-based energy modelling experiments on the BPI Challenge 2019 log.

The idea: take process variants from the log, treat each activity as a Markov state, hide made-up energy costs inside a synthetic signal, then try to recover those costs from the signal. On top of that, `markov_reward.py` uses the learned energies + transition probabilities to get expected energy per case.

This folder is self-contained — it reads the XES file directly and writes everything locally. The rest of the repo (`signal_generator/`, `Event Log Manager/`) is older stuff from the project I built on.

Data: [BPI Challenge 2019](https://www.tf-pm.org/resources/bpi-challenge/bpi-challenge-2019) (purchase-to-pay).

---

## how to run

```bash
cd thesis
pip install pm4py pandas numpy matplotlib networkx

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

Run `MarkovModel_clean.py` first. `markov_reward.py` only reads the CSVs it writes — no XES reload.

---

## what's in here

```
thesis/
├── BPI_Challenge_2019.xes          # input log (~695 MB)
├── MarkovModel_clean.py            # main script — use this one
├── markov_reward.py                # MRP layer on top
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

No requirements.txt in the repo — just pip install what you need.

Seeds are fixed (`SIGNAL_SEED = 123` etc.) so re-running on the same log gives the same numbers.
