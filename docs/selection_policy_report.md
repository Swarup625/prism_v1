# PRISM Alternative Model Selection Report

## Purpose

This experiment evaluates whether alternative baseline models can provide
useful evidence when the primary PRISM prediction may be unreliable.

The primary model is the frozen Random Forest baseline. Logistic Regression
and HistGradientBoosting are treated as alternative evidence sources.

## Dataset

- Validation states: 1,285
- Validation matches: 71
- Reserved 2025 test states: 1,268
- Reserved 2026 test states: 1,300

The 2025 and 2026 test datasets were not used for tuning or policy selection.

## Model Agreement

- Full agreement: 1,048 (81.56%)
- Disagreement: 237 (18.44%)

## Primary Baseline

- Model: random_forest
- Validation accuracy: 0.8054

## Alternative Consensus

An alternative prediction is proposed only when both alternative models
disagree with the Random Forest baseline and agree with each other.

- Alternative consensus states: 62
- Baseline wrong and alternative correct: 31
- Baseline correct and alternative wrong: 31

## Selection Policy

The dynamic selector does not automatically replace the primary prediction.

Instead, it provides evidence to the PRISM verifier, including:

1. Predictions from all baseline models.
2. Model confidence values.
3. Whether all models agree.
4. The number of models supporting each outcome.
5. An alternative prediction when both secondary models support the same outcome.

The final ACCEPT, CORRECT, or ABSTAIN decision will be made by the
verification module using this evidence together with reliability signals
and historical similarity evidence.

## Reproducibility

- Primary model: Random Forest
- Alternative models: Logistic Regression and HistGradientBoosting
- Validation data used for analysis only
- 2025 and 2026 test datasets remain untouched
- Total prediction runtime: 2.1636 seconds
