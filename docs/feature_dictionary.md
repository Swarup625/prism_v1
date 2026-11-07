# Feature Dictionary

# PRISM Feature Dictionary

## 1. Overview

This document describes the features used to represent an IPL match state.

Each row represents the state of a second-innings chase after a completed over.

All features are calculated using only information available at that point in the match.

---

## 2. Identification Features

### `match_id`

Unique identifier for the match.

This feature is used for tracking and dataset splitting. It should not be used as a predictive machine learning feature.

---

### `season`

The IPL season in which the match was played.

This field is primarily used to create the chronological dataset split.

---

### `over_completed`

The number of completed overs in the second innings at the time of prediction.

---

## 3. Match State Features

### `current_score`

The total runs scored by the chasing team at the end of the completed over.

---

### `wickets_lost`

The number of wickets lost by the chasing team at the end of the completed over.

---

### `target`

The number of runs required to win the match.

The target is derived from the first-innings score.

---

### `runs_required`

The number of additional runs required by the chasing team to reach the target.

The feature is calculated as:

```text
runs_required = target - current_score
```

---

### `balls_remaining`

The number of legal balls remaining in the innings.

For a standard 20-over innings:

```text
balls_remaining = 120 - legal_balls_completed
```

---

### `current_run_rate`

The scoring rate of the chasing team up to the current prediction point.

It is calculated as:

```text
current_run_rate
