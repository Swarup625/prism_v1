# PRISM Verification Decision Policy

## Purpose

The PRISM verifier combines the original baseline prediction with:

1. Error-risk estimation.
2. Historical similarity evidence.
3. Alternative-model evidence.

The verifier makes one of three decisions:

- **ACCEPT** — retain the original prediction.
- **CORRECT** — replace the original prediction with a supported alternative.
- **ABSTAIN** — return no definitive prediction.

## Selected Validation Policy

| Parameter | Selected Value |
|---|---:|
| Error-risk threshold | 0.50 |
| Similarity confidence threshold | 0.65 |
| Similarity support threshold | 0.50 |

## Decision Rules

### ACCEPT

The original baseline prediction is retained when:

- The estimated error risk is below the selected threshold, or
- The prediction is high-risk but no sufficiently supported alternative exists.

### CORRECT

A correction is made only when:

1. The prediction has high estimated error risk.
2. Historical similarity evidence strongly supports an alternative.
3. Alternative models provide consensus for an alternative.
4. Both evidence sources support the same alternative prediction.

High error risk alone is never sufficient to trigger a correction.

### ABSTAIN

PRISM abstains when:

- Error risk is high, and
- There is evidence against the baseline prediction,
- But the available correction evidence is conflicting or insufficient.

## Validation Results

| Metric | Value |
|---|---:|
| Baseline accuracy | 0.8054 |
| Final accuracy | 0.9060 |
| Coverage | 0.5214 |
| Abstention rate | 0.4786 |
| Selective risk | 0.0940 |
| Error detection rate | 0.7480 |
| Error recovery rate | 0.0240 |
| Harmful corrections | 0 |

## Experimental Protocol

All policy thresholds were selected using the validation dataset only.

The 2025 and 2026 test datasets were not used during verification
policy selection.
