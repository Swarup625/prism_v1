# Data Quality Report

# PRISM IPL Data Quality Report

## 1. Data Source

The IPL match data was obtained from Cricsheet in JSON format.

The source contains ball-by-ball information and match-level metadata for IPL matches.

The original dataset contained:

- **1,243 JSON match files**
- **295,732 delivery records**

All available JSON files were successfully parsed.

---

## 2. Parsed Dataset

The data parser produced two main datasets.

### Match-Level Dataset

The match-level dataset contains:

- `match_id`
- `date`
- `season`
- `team1`
- `team2`
- `venue`
- `winner`
- `result_type`

### Delivery-Level Dataset

The delivery-level dataset contains:

- `match_id`
- `innings`
- `over`
- `delivery_index`
- `actual_delivery`
- `batting_team`
- `batter`
- `non_striker`
- `bowler`
- `runs_batter`
- `runs_extras`
- `runs_total`
- `extras_type`
- `is_wicket`
- `wicket_count`

---

## 3. Parsing Results

The parsing process completed successfully.

| Item | Count |
|---|---:|
| JSON files found | 1,243 |
| Successfully parsed | 1,243 |
| Failed files | 0 |
| Matches | 1,243 |
| Delivery records | 295,732 |

---

## 4. Match Outcomes

The original dataset contained multiple result types.

| Result Type | Count |
|---|---:|
| Normal | 1,218 |
| Tie | 16 |
| No result | 9 |

The current PRISM experiment focuses only on normal-result matches because the prediction target is whether the chasing team wins the match.

Tied and no-result matches are excluded from the current dataset.

---

## 5. Selected Experimental Period

The PRISM experiment uses recent IPL seasons for temporal evaluation.

The selected period contains **487 matches** from the seasons:

- 2020/21
- 2021
- 2022
- 2023
- 2024
- 2025
- 2026

After filtering for normal results and matches with two innings, the final usable dataset contains:

**475 matches**

The remaining matches were excluded because they did not satisfy the requirements of the current prediction task.

---

## 6. Data Validation

The parsed and constructed datasets were audited for the following issues:

- Missing values.
- Invalid wickets.
- Negative scores.
- Negative runs required.
- Invalid target values.
- Invalid over numbers.
- Split inconsistencies.
- Duplicate over-level prediction states.
- Decreasing match scores.
- Decreasing wicket counts.
- Mathematical inconsistencies in runs required.
- Mathematical inconsistencies in balls remaining.

The final audit found no errors in these checks.

---

## 7. Final Dataset

The final match-state dataset contains:

| Item | Count |
|---|---:|
| Valid matches | 475 |
| Prediction states | 8,665 |

The dataset is used as the input for baseline machine learning models and the subsequent PRISM reliability pipeline.

---

## 8. Reproducibility

The data preparation process can be reproduced using:

```text
src/data/parse_cricsheet.py
src/data/build_states.py
scripts/audit_match_states.py
```

The processed data is stored separately from source code to avoid unnecessary duplication and large file uploads to the repository.

---

## 9. Limitations

The current experiment has several limitations:

1. Only IPL matches are considered.
2. Only second-innings match states are used.
3. Tied and no-result matches are excluded.
4. The dataset represents historical IPL conditions and may not capture all future changes in teams, venues, or playing strategies.

These limitations will be considered when interpreting