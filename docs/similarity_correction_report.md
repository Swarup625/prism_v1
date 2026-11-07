# PRISM Historical Similarity Corrector

## Purpose

The similarity corrector provides an independent source of evidence by
finding historical IPL match situations that resemble a current match
state.

It does not modify the original baseline prediction directly. Instead,
it produces an alternative prediction together with evidence about the
historical situations supporting that prediction.

## Leakage-Safe Design

Historical reference data contains only the training split.

- Training reference states: 4,812
- Training matches: 262
- Validation states: 1,285
- Validation matches: 71

Validation data was used only to select the neighborhood size.

The reserved 2025 and 2026 test sets were not used.

## Similarity Features

The following match-state variables were standardized before computing
Euclidean distances:

- `current_score`
- `wickets_lost`
- `runs_required`
- `balls_remaining`
- `current_run_rate`
- `required_run_rate`
- `runs_last_3_overs`
- `wickets_last_3_overs`

## Selected Configuration

- Neighborhood size (K): 50
- Selection criterion: Highest validation ROC-AUC
- Tuning runtime: 0.1001 seconds

## Validation Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.7891 |
| Precision | 0.7802 |
| Recall | 0.7914 |
| F1 Score | 0.7858 |
| ROC-AUC | 0.8837 |
| Log Loss | 0.4464 |

## Evidence Produced

For every validation prediction, the system records:

- Similarity-based win probability
- Alternative predicted class
- Number of historical neighbors
- Historical support for chasing-team wins and losses
- Mean, minimum, and maximum neighbor distances

## Role in PRISM

The similarity corrector is an evidence source for the later PRISM
verification stage. A disagreement between the baseline model and
historical similarity does not automatically trigger a correction.

A later verification policy will combine this evidence with error-risk
signals and alternative model predictions before deciding whether to:

- ACCEPT the original prediction
- CORRECT the prediction
- ABSTAIN from making a definitive prediction

## Limitations

The current implementation uses Euclidean distance in a standardized
feature space. Similarity quality may depend on feature selection,
distance weighting, and the choice of neighborhood size. These settings
should remain frozen before final testing.
