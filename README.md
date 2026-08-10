# KAUST Kaggle Challenge 3: Offset-Fault Detection in Low-Cost Air Quality Sensors

**24th place** out of the competition field, private F1 **0.972**, with 7 submissions over a single working day.

The result I actually care about is a different one: a model using **one engineered feature** scored **0.977** on the private leaderboard, beating every 51-feature model I built. I did not select it. This README explains what the feature was, why it won, and why I talked myself out of it.

Kaggle notebook: [Stage 3, few features](https://www.kaggle.com/code/ghostylicious/kaust-kaggle-challenge-3-2026-stage-3-few-features)

---

## The problem

Two low-cost PM2.5 sensors sit on the same pole. One of them intermittently develops an **offset fault**, drifting away from the true concentration. Given a timestamped reading from both sensors plus temperature and relative humidity, decide whether that row is faulty.

| | |
|---|---|
| Task | Binary classification |
| Metric | F1 |
| Train | 141,538 rows, 15 Oct to 30 Nov 2021 |
| Test | 155,639 rows, 1 Dec 2021 to 21 Jan 2022 |
| Class balance | 36.8% faulty |
| Leaderboard split | 10% public, 90% private |
| Budget | 8 submissions, 2 selected |
| Provided baseline | Random Forest, 96.2% CV / 92.2% public LB |
| Constraints | No external data, no deep learning |

Train ends exactly where test begins, so this is forecasting, not a random split. There was also a second prize pool for the **best model using three features or fewer**, which is where most of the interesting work went.

## What the data actually turned out to be

Before writing a model I spent the first hour trying to falsify the assumptions in my own plan. Three of them were wrong.

**Faults are isolated single rows, not episodes.** Median run length of the fault class is 1. There are 101,257 label runs across 141,538 rows. The fault rate is flat at roughly 0.37 on every one of the 47 days and in every hour of the day. So "how long does a fault last" and "does the fault rate drift" have no answer to find, and any model built around fault duration is chasing nothing.

That single fact is the whole competition. **Because faults hit isolated rows, a row's temporal neighbours are a clean estimate of what the reading should have been.** A faulty row sits far below its own local level on both sensors, and much further below on Sensor 2 than on Sensor 1. That asymmetry is also why the faulty class shows a steeper Sensor1-vs-Sensor2 line: S2 gets pulled down harder.

**Test timestamps are minute-resolution; train timestamps are second-resolution.** Every test timestamp has its seconds field zeroed. Both files are the same underlying stream at about 2.18 rows per minute. Any order-dependent feature (lags, neighbours, rolling windows) can therefore be built more precisely on train than it can ever be built on test. Sorting train by its exact second and test by minute inflates local validation by measuring a precision that does not exist at inference time. Every notebook here floors **both** tables to the minute and breaks within-minute ties with the same seeded random rule.

**The "2022" hint was a red herring.** The organisers dropped a meme implying something special about 2022. January 2022 turns out to have the same fault-proxy share as December 2021 and as the training period. A hard rule mapping 2022 to the healthy class would have destroyed roughly 40% of the positive class on a 90% private split. The defensible reading is the boring one: training contains a single year, so raw calendar features are constant during training and unseen at inference. Adding them measurably **hurt**, dropping future-holdout F1 from 0.939 to 0.921.

**Temperature and humidity are a trap.** Adversarial validation separating train from test reached AUC 0.952, driven almost entirely by those two columns as autumn turned into winter. Dropping them improved every model and halved fold-to-fold variance.

## Feature stages

Model settings were frozen across stages so every score change is attributable to the feature family and not to tuning.

| Stage | Features | Strict future holdout F1 |
|---|---|---|
| Raw sensors + weather | 4 | 0.9297 |
| + Sensor1/Sensor2 relationship | 12 | 0.9327 |
| + calendar | 17 | 0.9166 |
| **+ temporal neighbours** | 33 | **0.9946** |
| + rolling aggregates | 63 | 0.9950 |
| Drop weather and calendar | 51 | 0.9957 |

The neighbour family is the step change. Everything before it is noise around the provided baseline; everything after it is decoration.

## The one feature

The best single column, and the thing I would take away from this whole competition:

```python
# minute-floored, chronologically sorted stream of train + test
N = pd.concat([S2.shift(j) for j in range(1, 6)] +
              [S2.shift(-j) for j in range(1, 6)], axis=1)

S2_z_K5 = (S2 - N.mean(axis=1)) / (N.std(axis=1) + 1e-3)
```

Sensor 2, z-scored against its five nearest neighbours on each side. That is it. No labels are used, so it is computable identically on the test set.

Alone, it scored **0.99369** on forward-chaining validation and **0.977 on the private leaderboard**, which was better than every multi-feature model I built.

Things I tried that did **not** beat it: robust median/MAD z-scores, log-ratios to the neighbour median, upper-quantile ambient estimates (on the theory that 37% faulty neighbours bias the median downward), ambient estimated from the two-sensor mean, time-window versions immune to within-minute ordering, and neighbourhood widths of 3, 4, 6, 8, 12 and 20. K=5 with a plain mean/std z-score won every comparison. Greedy forward selection from it was useless: adding a second feature moved the score from 0.9923 to at best 0.9927, with most candidates making it worse.

Extra model capacity did not help either. Depth 8 at 800 iterations, depth 10 at 1200, and depth 10 at 2000 all scored at or below plain depth 6 at 250 iterations. The three-feature ceiling is an information limit, not a capacity limit.

## The mistake

Here is the part worth writing down.

For the fewest-features slot I had two candidates: the single feature above, and a trio adding Sensor 1's z-score and the symmetric sensor difference. Two validation schemes disagreed:

| Candidate | Forward chaining | Strict future holdout | Date-grouped CV |
|---|---|---|---|
| Solo (1 feature) | 0.99369 | **0.99325** | 0.94824 |
| Trio (3 features) | 0.99381 | 0.99171 | **0.95906** |

The trio won date-grouped CV by 0.011, seven times larger than the margin by which the solo won the strict holdout. I reasoned that since the test set sits two months beyond training, the scattered-days scheme was the better proxy for a regime shift, and picked the trio.

That reasoning was wrong, and it was wrong in a way I should have caught. The **strict future holdout is the only scheme whose shape matches the actual problem**: train on the past, predict forward. Date-grouped CV lets a model see days on both sides of every held-out day, which is not a harder version of the real task, it is a different task. When two schemes disagree, the one that structurally resembles the test set wins, and I had written exactly that sentence in an earlier notebook before overriding it.

The private leaderboard was blunt about it:

| Submission | Public | Private |
|---|---|---|
| Solo, 1 feature | 0.992 | **0.977** |
| Neighbour model, date-grouped threshold | 0.995 | 0.977 |
| Full 51-feature model | 0.996 | 0.972 |
| Order-averaged full model (selected) | 0.996 | 0.972 |
| **Trio, 3 features (selected)** | 0.992 | **0.960** |

The trio was the single worst submission I made. The solo was the best. Selecting it would have scored 0.977 and moved me roughly ten places up the board.

There is a second lesson underneath the first. Everything at public 0.996 collapsed to private 0.971 or 0.972, while the 0.992 public models held at 0.977. **The public leaderboard ranking was inverted.** With only 10% public and 8 submissions, the public column was never going to discriminate between candidates that disagreed on 250 of 155,639 rows, and I knew that, and I still let the tight cluster at 0.996 make me feel safer about the complicated models than about the simple one.

## Repository layout

```
notebooks/
  stage1_date_grouped_cv.ipynb      date-grouped GroupKFold, ablations, adversarial validation
  stage2_strict_temporal.ipynb      forward chaining, honest thresholds, ordering-trap measurement
  stage3_few_features.ipynb         the fewest-features track, published on Kaggle
  stage4_robustness.ipynb           neighbourhood-width sweep and ordering robustness
metrics/                            per-stage F1, fold spreads, thresholds, confusion counts
submissions/                        all 7 candidates
images/                             submissions screenshot
```

## What I would do differently

1. Pick the validation scheme that structurally matches the test split, and do not let a second scheme with a bigger margin override it.
2. Keep one final slot for the simplest model that validates well. The simplest model was the most robust model here, and that is not a coincidence: fewer parameters means fewer things to miscalibrate when the regime moves.
3. Stop treating a tight public-leaderboard cluster as information. It is noise with a decimal point.

The feature engineering held up. The judgement about which model to trust did not.

## Rules compliance

No external data, no external labels, no weather data, no deep learning. Features were built on train and test concatenated, which uses no labels and only gives a test row the same kind of temporal neighbourhood a training row gets.
