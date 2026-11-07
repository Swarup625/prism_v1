# Data

# PRISM Dataset Documentation

## 1. Overview

This document describes the dataset used for the initial PRISM experiments.

The dataset is constructed from IPL ball-by-ball match data and converted into over-level prediction states for the second innings of each match.

The objective is to predict whether the chasing team will eventually win.

---

## 2. Data Source

The original data consists of IPL match records in JSON format.

The source provides:

- Match metadata.
- Teams.
- Dates.
- Venues.
- Innings.
- Ball-by-ball deliveries.
- Runs.
- Wickets.
- Match outcomes.

---

## 3. Dataset Construction

A single machine learning example is created after every completed over of the second innings.

For example, a 20-over chase may produce prediction states after:

```text
Over 1
Over 2
Over 3
...
Over 19
```

The system does not use future deliveries when constructing a state.

---

## 4. Dataset Size

The final dataset contains:

- **475 matches**
- **8,665 prediction states**

The number of states per match can vary because some matches finish before the full 20 overs.

---

## 5. Target Variable

The target variable is:

```text
chasing_team_won
```

Its values are:

- `1`: The team batting second eventually wins.
- `0`: The team batting second does not win.

Only normal-result matches are included.

---

## 6. Dataset Splits

The dataset uses chronological season-based splits.

| Split | Seasons | Purpose |
|---|---|---|
| Training | 2020/21–2023 | Train machine learning models |
| Validation | 2024 | Select models and thresholds |
| Test 2025 | 2025 | Final temporal evaluation |
| Test 2026 | 2026 | Final temporal evaluation |

All prediction states from a single match remain in the same split.

---

## 7. Split Statistics

| Split | Matches | Prediction States |
|---|---:|---:|
| Training | 262 | 4,812 |
| Validation | 71 | 1,285 |
| Test 2025 | 70 | 1,268 |
| Test 2026 | 72 | 1,300 |

The 2025 and 2026 test sets will be evaluated separately and may also be reported together for an overall summary.

---

## 8. Data Storage

The main processed files are:

```text
data/processed/matches.csv
data/processed/deliveries.parquet
data/match_states.parquet
data/split_manifest.csv
```

The source code used to construct the dataset is stored separately from the generated data.

---

## 9. Usage Rules

All experiments using this dataset must follow these rules:

1. Do not randomly mix states from the same match across splits.
2. Do not use 2025 or 2026 data for model selection.
3. Use validation data to select hyperparameters and decision thresholds.
4. Do not construct features using future deliveries.
5. Record any additional filtering or transformations applied during experiments.