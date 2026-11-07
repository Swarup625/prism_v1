# Leakage Audit

# PRISM Data Leakage Audit

## 1. Purpose

This document records the checks performed to reduce the risk of information leakage in the PRISM dataset.

Information leakage occurs when a model receives information that would not be available at the time a prediction is made.

For this project, the prediction is made after each completed over of the second innings.

---

## 2. Future Delivery Check

Each prediction state is constructed using only deliveries that occurred on or before the completed over.

The following information is not used when constructing the current state:

- Future runs.
- Future wickets.
- Future overs.
- Final second-innings score.
- Future batting events.
- Future bowling events.

The final match outcome is stored only as the target label.

---

## 3. Target Leakage Check

The feature matrix and target variable are kept conceptually separate.

The target variable is:

```text
chasing_team_won
```

This value represents the final outcome of the match and must never be used as an input feature during baseline model training.

---

## 4. Split Leakage Check

The dataset is split by match and season.

All prediction states from one match belong to exactly one dataset split.

The audit confirmed that no match appears in multiple splits.

This prevents the model from training on some overs from a match and being evaluated on other overs from the same match.

---

## 5. Temporal Leakage Check

The dataset uses chronological splits:

- Training: 2020/21–2023.
- Validation: 2024.
- Final test: 2025.
- Final test: 2026.

Later seasons are not used to train earlier-stage models.

The final test seasons must not be used to select:

- Features.
- Model hyperparameters.
- Detection thresholds.
- Correction thresholds.
- Verification policies.

---

## 6. Mathematical Consistency Checks

The match-state dataset was audited for consistency.

The following checks passed:

- No negative scores.
- No negative wickets.
- No invalid wicket counts.
- No negative runs required.
- No negative balls remaining.
- No invalid target values.
- No invalid over values.
- Runs required values are mathematically consistent.
- Balls remaining values are mathematically consistent.
- Match scores progress monotonically.
- Wicket counts progress monotonically.
- No duplicate over-level states were found.

---

## 7. Model Development Risk

Additional leakage can occur during later stages of PRISM.

The following rules will therefore be enforced:

1. Baseline model predictions used to train an error detector must be generated in a leakage-safe manner.
2. Reliability thresholds must be selected using validation data.
3. Similarity-based correction must retrieve examples from the appropriate historical training data.
4. Alternative model selection must not use final test outcomes to choose a model.
5. Final PRISM evaluation must use frozen models and thresholds.

---

## 8. Current Conclusion

The current match-state dataset passed the implemented data-quality and leakage checks.

Future PRISM modules must continue to follow the same separation between:

- Training data.
- Validation data.
- Final test data.

The leakage audit will be updated if new features or data-processing procedures are introduced.
