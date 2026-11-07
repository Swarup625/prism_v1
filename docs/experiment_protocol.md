# Experiment Protocol
# PRISM Experiment Protocol

## 1. Purpose

This document defines the experimental procedure for evaluating PRISM.

The purpose is to ensure that all models and PRISM components are evaluated fairly, reproducibly, and without information leakage.

---

## 2. Prediction Task

The task is binary classification.

At the end of every completed over during the second innings, the system predicts whether the chasing team will eventually win the match.

Target:

- `1` — Chasing team wins.
- `0` — Chasing team does not win.

---

## 3. Dataset Split

The experiment uses a chronological split.

| Dataset | Seasons | Purpose |
|---|---|---|
| Training | 2020/21–2023 | Model training |
| Validation | 2024 | Model and threshold selection |
| Test | 2025 | Final evaluation |
| Test | 2026 | Final evaluation |

The two test seasons must remain untouched during model development.

---

## 4. Experimental Stages

### Stage 1: Baseline Models

Train multiple baseline machine learning models using only the training dataset.

Candidate models may include:

- Logistic Regression.
- Random Forest.
- Gradient Boosting.
- XGBoost or another suitable boosting method.

Model selection is performed using validation performance.

---

### Stage 2: Reliability Analysis

After generating baseline predictions, calculate reliability signals such as:

- Prediction confidence.
- Probability margin.
- Predictive entropy.
- Calibration quality.
- Model disagreement.

The purpose is to determine whether these signals are associated with incorrect predictions.

---

### Stage 3: Error Detection

Train a detector that estimates whether a baseline prediction is likely to be incorrect.

The detector must be trained using leakage-safe predictions.

Thresholds must be selected using validation data.

---

### Stage 4: Candidate Correction

Generate alternative predictions using:

1. Similar historical match states.
2. Alternative machine learning models.

Candidate corrections must be evaluated separately from the final verification decision.

---

### Stage 5: Verification and Abstention

The verification module decides whether to:

- Accept the original prediction.
- Correct the prediction.
- Abstain.

Decision thresholds must be selected using validation data only.

---

## 5. Final Evaluation

The following configurations will be compared:

1. Baseline model only.
2. Baseline model with error detection.
3. Baseline model with similarity-based correction.
4. Baseline model with alternative-model correction.
5. Full PRISM without verification.
6. Full PRISM.
7. Full PRISM without abstention.

---

## 6. Evaluation Rules

The following rules apply to every experiment:

- Use the fixed dataset split.
- Keep all states from one match in the same split.
- Do not use future information.
- Do not tune on final test data.
- Record random seeds where applicable.
- Record model configurations.
- Keep 2025 and 202
