# PRISM Claim-Evidence Table

| claim                                                           | evidence                                                               | support                           |
|:----------------------------------------------------------------|:-----------------------------------------------------------------------|:----------------------------------|
| PRISM reduces selective prediction risk on test_2025            | Baseline risk = 0.1924; PRISM risk = 0.1379                            | Supported                         |
| PRISM improves reliability on returned predictions on test_2025 | Baseline accuracy = 0.8076; PRISM accuracy = 0.8621; coverage = 0.5718 | Supported with coverage trade-off |
| PRISM reduces selective prediction risk on test_2026            | Baseline risk = 0.2454; PRISM risk = 0.0853                            | Supported                         |
| PRISM improves reliability on returned predictions on test_2026 | Baseline accuracy = 0.7546; PRISM accuracy = 0.9147; coverage = 0.5231 | Supported with coverage trade-off |
| Alternative-model selection consistently improves predictions   | Ablation results show inconsistent improvement across test splits.     | Not supported                     |
| Abstention contributes to PRISM reliability                     | Full PRISM has lower selective risk than the no-abstention variant.    | Supported                         |
