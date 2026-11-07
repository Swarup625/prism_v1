# PRISM Final Results Summary
## Experimental Protocol
All final experiments used frozen trained models, unchanged decision thresholds, and historical similarity references constructed from training data only. No models were retrained and no thresholds were tuned using the 2025 or 2026 test sets.
## Baseline and Full PRISM Results
### test_2025
- Baseline accuracy: 0.8076
- Full PRISM accuracy on returned predictions: 0.8621
- Accuracy difference: +0.0545
- PRISM coverage: 0.5718
- Baseline risk: 0.1924
- PRISM selective risk: 0.1379
- Selective risk reduction: +0.0545
### test_2026
- Baseline accuracy: 0.7546
- Full PRISM accuracy on returned predictions: 0.9147
- Accuracy difference: +0.1601
- PRISM coverage: 0.5231
- Baseline risk: 0.2454
- PRISM selective risk: 0.0853
- Selective risk reduction: +0.1601

## Ablation Findings
The ablation study shows that the individual correction components do not provide uniform improvements across all test seasons.
Similarity-based correction improved the baseline on both test sets, whereas alternative-model selection produced smaller or inconsistent gains.
The full PRISM system achieved its strongest reliability results when verification and abstention were enabled, indicating that selectively withholding unsupported predictions is an important component of the overall reliability mechanism.

## Negative Results and Limitations
- Alternative-model selection did not consistently improve performance over the baseline.
- PRISM does not return a prediction for every match state; the reliability improvement is accompanied by reduced coverage.
- Some corrections can still be harmful, demonstrating that correction evidence should not be treated as perfectly reliable.
- Results are based on IPL match-state data and may not directly generalize to other sports prediction tasks.
