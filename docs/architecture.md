# Architecture

# PRISM System Architecture

## 1. Overview

PRISM stands for **Prediction Reliability and Intelligent Self-Correction Mechanism**.

The goal of PRISM is not only to generate a machine learning prediction, but also to evaluate whether that prediction can be trusted. When a prediction appears unreliable, PRISM investigates possible alternatives and decides whether to retain the original prediction, apply a correction, or abstain from making a confident decision.

The system follows the pipeline:

**Predict → Detect → Diagnose → Correct → Verify → Accept, Correct, or Abstain**

The first implementation of PRISM uses IPL cricket match prediction as an experimental case study.

---

## 2. Prediction Task

The machine learning task is to predict whether the team batting in the second innings will eventually win the match.

A prediction state is created after every completed over of the second innings.

For each prediction state:

- **Input:** Match information available up to that completed over.
- **Target:** Whether the chasing team eventually wins the match.
- **Prediction type:** Binary classification.

The target variable is defined as:

- `1` — The chasing team wins.
- `0` — The chasing team does not win.

Only normal-result matches with two innings are used in the current experiment.

---

## 3. PRISM Pipeline

### Stage 1: Baseline Prediction

A machine learning model receives the current match state and produces:

- A predicted class.
- A probability or confidence score.

This prediction represents the original answer that PRISM will evaluate.

---

### Stage 2: Reliability Signal Generation

PRISM calculates signals that may indicate whether the original prediction is reliable.

Examples include:

- Prediction confidence.
- Probability margin.
- Predictive entropy.
- Model disagreement.
- Similarity to historical match states.

These signals provide evidence for the next stage.

---

### Stage 3: Error Detection

The error detection module estimates whether the original prediction is likely to be incorrect.

The module produces:

- An error risk score.
- A decision about whether the prediction should receive further review.

A high-risk prediction is not automatically corrected. It is instead passed to the correction and verification stages.

---

### Stage 4: Candidate Correction

When a prediction requires further investigation, PRISM can generate alternative predictions.

The initial system investigates two correction mechanisms:

1. **Similarity-based correction**
   - Finds historical match states similar to the current state.
   - Uses the outcomes of similar situations as supporting evidence.

2. **Alternative model selection**
   - Compares predictions from multiple machine learning models.
   - Investigates whether another model provides a more reliable alternative.

---

### Stage 5: Verification

The verification module receives:

- The original prediction.
- Reliability signals.
- Error risk score.
- Alternative predictions.
- Supporting evidence.

It evaluates whether there is sufficient evidence to change the original prediction.

---

### Stage 6: Final Decision

PRISM produces one of three decisions:

- **ACCEPT** — Retain the original prediction.
- **CORRECT** — Replace the original prediction with a supported alternative.
- **ABSTAIN** — Do not provide a definitive prediction when the available evidence is insufficient.

---

## 4. System Components

The repository is organized into the following major modules:

```text
src/
├── data/           # Data parsing and match-state construction
├── baseline/       # Baseline machine learning models
├── signals/        # Reliability and uncertainty signals
├── detection/      # Error detection
├── correction/     # Candidate correction methods
├── verification/   # Final decision and abstention
└── evaluation/     # Metrics and robustness experiments
```

The main integrated system will be managed through:

```text
src/prism.py
scripts/run_prism.py
```

---

## 5. Design Principles

The PRISM architecture follows several important principles:

1. **No future information:** Every prediction uses only information available at the prediction time.
2. **Separation of validation and testing:** Final test data must not be used to select models or thresholds.
3. **Traceable decisions:** Every correction or abstention should have supporting evidence.
4. **Safe correction:** A prediction should not be changed unless there is sufficient evidence that the alternative is more reliable.
5. **Honest abstention:** Abstaining is treated as an unresolved prediction, not as a correct prediction.
6. **Reproducibility:** Experiments should record data splits, configurations, random seeds, and evaluation procedures.

---

## 6. Current Implementation Status

The data preparation stage has been completed.

The current dataset contains:

- 475 valid IPL matches from the selected seasons.
- 8,665 over-level prediction states.
- Training data from 2020/21 to 2023.
- Validation data from 2024.
- Separate final test sets for 2025 and 2026.

The next stage is to train baseline machine learning models that will provide the original predictions for the PRISM pipeline.