# AGENTS.md — PRISM

ML research project: 5-stage self-correction pipeline (Detect → Diagnose → Correct → Verify → Decide) for IPL second-innings win prediction. Python 3.11+, no packaging config (no pyproject/setup); imports are relative to repo root.

## Entry points

- Pipeline demo: `python scripts/run_prism.py`
- Zero-dependency web server (stdlib only, serves `/api/*` + frontend): `python backend/server.py 8000` (or `start_server.bat` on Windows)
- Flask API server (needs deps): `python backend/app.py`; deps from `backend/requirements.txt` (Flask/Gunicorn/sklearn) — root `requirements.txt` is the separate ML-research set.
- Authoritative frozen evaluation: `src/evaluation/final_evaluation.py` — must stay training-free and cannot tune thresholds on test data.

## Environment gotchas

- Does not need ML deps. `src/prism.py` and both servers degrade to "Standalone Engine" heuristic mode when numpy/sklearn are missing; they import heavy deps lazily inside try/except. Keep that pattern in new code.
- Parquet I/O (`data/processed/*.parquet`, `data/match_states.parquet`) needs `pyarrow`, which is **not** listed in either requirements file — install it separately or load the models path will fall back to heuristics.
- Frozen models live in `models/*.joblib`; script entrypoints self-inject the project root into `sys.path`, so imports are always `from src.x import y`.

## Testing

- `python -m unittest discover tests` — tests are unittest, not pytest (pytest is in requirements but unused).
- Only `tests/test_pipeline.py` has real tests (4, run in seconds). Every other `tests/**/test_*.py` is a one-line placeholder — don't count them as coverage.
- Tests pass with zero ML deps installed (heuristic fallback path is what's exercised).

## Data pipeline (order matters)

`src/data/`: `parse_cricsheet.py` → `clean_matches.py` → `build_states.py` → `build_features.py` → `split_matches.py`

Writing to `data/raw/ipl_json`, then `data/processed/`, then `data/match_states.parquet` + `data/split_manifest.csv` at repo root of `data/`.

**Frozen split — do not change:** Train 2020/21–2023, Validation 2024, Test 2025–2026. All snapshots from one match stay in one split (zero intra-match leakage). Never tune thresholds/hyperparameters on the test seasons; `results/final_predictions_test_*` are the frozen test outputs.

## Config files are mostly inert

`configs/*.yaml` is reference documentation; only `configs/failure_scenarios.yaml` is actually loaded (by `src/evaluation/robustness_tests.py`). The frozen decision policy/thresholds live in code: `src/verification/decision_policy.py`.

## Backend duplication

`backend/app.py` (Flask) and `backend/server.py` (stdlib) are parallel implementations of the same `/api/*` endpoints and pipeline-stage metadata. Any endpoint or metadata change must be applied to both files or one server mode goes stale.

## Frontend / deploy

- `frontend/index.html` is the whole SPA (Tailwind CDN + Chart.js, no build step). `docs/index.html` is a byte-identical mirror — after editing one, update the other.
- GitHub Pages CI (`.github/workflows/deploy.yml`) deploys **only** `frontend/`; Netlify publishes `frontend/`, Vercel rewrites everything to it. No lint, typecheck, or test CI exists — only the Pages deploy on `main`.
- README references `scripts/run_robustness.py`, which doesn't exist; robustness runner is `src/evaluation/robustness_tests.py`.
