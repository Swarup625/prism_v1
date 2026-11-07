# PRISM Baseline Model Calibration Report

## Purpose

This analysis evaluates whether the probabilities produced by the selected
baseline model correspond to the actual frequency of the predicted outcome.

The evaluation uses the validation dataset only.

## Dataset

- Total validation predictions: **1285**
- Model: **random_forest**

## Calibration Metrics

| Metric | Value |
|---|---:|
| Brier Score | 0.141309 |
| Expected Calibration Error (ECE) | 0.054395 |

### Interpretation

A lower Brier Score indicates better probability predictions.

A lower Expected Calibration Error indicates that predicted probabilities
are closer to the actual observed outcome frequencies.

## Calibration by Probability Range

| Probability Range | Predictions | Mean Predicted Probability | Observed Win Rate | Calibration Gap |
|---|---:|---:|---:|---:|
| (-0.001, 0.1] | 160 | 0.0566 | 0.0312 | 0.0253 |
| (0.1, 0.2] | 203 | 0.1471 | 0.1478 | -0.0007 |
| (0.2, 0.3] | 162 | 0.2491 | 0.2099 | 0.0393 |
| (0.3, 0.4] | 122 | 0.3475 | 0.3607 | -0.0132 |
| (0.4, 0.5] | 106 | 0.4533 | 0.5660 | -0.1127 |
| (0.5, 0.6] | 92 | 0.5464 | 0.7500 | -0.2036 |
| (0.6, 0.7] | 104 | 0.6455 | 0.7308 | -0.0852 |
| (0.7, 0.8] | 78 | 0.7515 | 0.9103 | -0.1588 |
| (0.8, 0.9] | 120 | 0.8491 | 0.8833 | -0.0342 |
| (0.9, 1.0] | 138 | 0.9513 | 0.9638 | -0.0124 |


## Conclusion

The calibration analysis provides an additional view of prediction
reliability. Calibration information can be used alongside confidence,
probability margin, and predictive entropy when studying whether a
prediction is likely to be unreliable.

This analysis does not use the 2025 or 2026 final test datasets.
