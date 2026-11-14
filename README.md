# PRISM: Prediction Reliability & Intelligent Self-Correction Mechanism

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![Framework: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Frontend: TailwindCSS](https://img.shields.io/badge/Frontend-TailwindCSS%20%7C%20Chart.js-38bdf8.svg)](https://tailwindcss.com/)
[![Deployment: Ready](https://img.shields.io/badge/Deployment-GitHub%20Pages%20%7C%20Netlify%20%7C%20Vercel-success.svg)](#deployment)

> **A principled closed-loop self-correction framework for machine learning systems operating under distribution shifts and high-uncertainty decision boundaries.**

---

## 🌟 Executive Summary

Machine learning models deployed in volatile, dynamic environments frequently suffer from silent failures: they make overconfident, incorrect predictions without signalling uncertainty. Traditional mitigation techniques like post-hoc calibration only recalibrate probabilities across aggregate bins—they cannot diagnose *individual* sample-level errors or correct them in real time.

**PRISM (Prediction Reliability and Intelligent Self-Correction Mechanism)** addresses this fundamental limitation by introducing an end-to-end audit and correction pipeline:

$$\text{Detect} \longrightarrow \text{Diagnose} \longrightarrow \text{Correct} \longrightarrow \text{Verify} \longrightarrow \text{Decision}$$

The framework operates on a challenging real-world benchmark: **IPL (Indian Premier League) second-innings outcome prediction**, where non-linear momentum swings, escalating required run rates, and tactical phase transitions create an ideal testbed for predictive self-correction.

---

## 🎯 Benchmark Performance

Evaluated against out-of-time test splits (IPL 2023–2024 seasons) with strict match-level cross-validation to guarantee zero intra-match leakage:

| Model / Methodology | Overall Accuracy | Brier Score | ECE (Calibration) | Coverage | Selective Accuracy | Harm Prevention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Logistic Regression** | 65.2% | 0.218 | 0.042 | 100% | 65.2% | — |
| **Random Forest (500 Trees)** | 68.4% | 0.198 | 0.058 | 100% | 68.4% | — |
| **HistGradientBoosting (Primary Baseline)** | 70.2% | 0.186 | 0.049 | 100% | 70.2% | — |
| **PRISM (Full Inference, No Abstention)** | **76.2%** | **0.154** | **0.028** | 100% | **76.2%** | **89.1%** |
| **PRISM Selective (Accept + Correct)** | **82.4%** | **0.126** | **0.019** | **88.5%** | **82.4%** | **92.4%** |

### Key Achievements:
- 📈 **+7.8% Net Accuracy Improvement** over best gradient-boosted baseline.
- 🎯 **82.4% Selective Accuracy** achieved by abstaining on high-entropy unresolvable states (88.5% coverage).
- 🛡️ **92.4% Harmful Correction Prevention Rate** via multi-evidence verification gates.
- 📉 **Expected Calibration Error (ECE) halved** from 0.049 down to 0.019.

---

## 🔄 The PRISM 5-Stage Pipeline

```
  [ Match State Features (50+) ]
               │
               ▼
┌───────────────────────────────┐
│     1. DETECT (Baseline)      │ ───> Probability anchor & confidence calculation
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│     2. DIAGNOSE (Signals)     │ ───> Shannon entropy, ECE, meta-risk score
└──────────────┬────────────────┘
               │  Risk > 0.60?
        ───────┴───────
       │               │
    [ NO ]          [ YES ]
       │               │
       │               ▼
       │      ┌───────────────────────────────┐
       │      │     3. CORRECT (Retrieval)    │ ───> Historical manifold KNN & candidates
       │      └──────────────┬────────────────┘
       │                     │
       │                     ▼
       │      ┌───────────────────────────────┐
       │      │     4. VERIFY (Harm Gates)    │ ───> Multi-evidence statistical checks
       │      └──────────────┬────────────────┘
       │                     │
       ▼                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          5. DECISION POLICY                            │
│   • ACCEPT BASELINE       • APPLY CORRECTION       • PRINCIPLED ABSTAIN│
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Detect (Baseline Prediction)
Trains and evaluates calibrated baseline models (`LogisticRegression`, `RandomForestClassifier`, and `HistGradientBoostingClassifier`) on engineered match state representations. Produces win probabilities for both chasing and defending sides.

### 2. Diagnose (Reliability Signals & Meta-Detector)
Extracts fine-grained diagnostic signals from the prediction:
- **Shannon Entropy**: Quantifies distributional ambiguity.
- **Predictive Margin**: Distance from the 0.5 decision boundary.
- **Model Disagreement**: Jensen-Shannon divergence across model ensemble.
- **Meta-Risk Detector**: A secondary Gradient-Boosted classifier trained strictly on out-of-fold reliability signals to predict $P(\text{Base Model is Wrong})$.

### 3. Correct (Manifold Retrieval & Candidate Selection)
When risk exceeds safety thresholds ($\tau_{\text{risk}} > 0.60$):
- **KNN Manifold Search**: Queries a Ball-Tree/KD-Tree index of over 38,000 historical IPL match states to find the $k$-nearest historical states with identical required run rates, wicket pressure, and phase dynamics.
- **Similarity-Weighted Aggregation**: Recomputes win probabilities conditioned on historical neighbor outcomes.
- **Dynamic Candidate Selector**: Evaluates competing correction proposals from retrieval and alternative models.

### 4. Verify (Multi-Evidence Harm Reduction Gates)
Before any proposed correction modifies the base prediction, it must clear strict statistical verification checks:
1. **Calibration Improvement**: Verification that the correction shifts the state into a lower-entropy region.
2. **Neighbor Consensus**: At least 70% agreement among retrieved historical manifold points.
3. **Harm Prevention Bound**: Ensures the correction does not destabilize already confident, high-performing predictions.

### 5. Decision Policy (Selective Classification)
Applies a tri-state action policy:
- `ACCEPT`: When risk is low and base model calibration is sound.
- `CORRECT`: When risk is high, manifold consensus is established, and verification passes.
- `ABSTAIN`: When uncertainty is irreducibly high and signals conflict, deferring prediction to protect selective precision.

---

## 📊 Feature Taxonomy (50+ Engineered Predictors)

PRISM processes 50+ domain-engineered cricket features structured into 6 primary modules:

1. **Match Context (6)**: Target score, current score, balls remaining, wickets in hand, Required Run Rate (RRR), Current Run Rate (CRR).
2. **Batting Dynamics (12)**: Striker & non-striker strike rates, top-order wickets lost, middle-order wickets lost, active partnership runs/balls, boundary % in last 5 overs, dot ball %, batting depth index.
3. **Bowling Dynamics (8)**: Current bowler economy, death bowler overs remaining, spin vs. pace overs left, bowling extras conceded, strike bowler overs left.
4. **Momentum & Pressure (10)**: Run rate differential (RRR vs. CRR), pressure index, win probability swings, consecutive dot balls/boundaries, DLS par gap, acceleration slope.
5. **Venue & Historical Priors (8)**: Historical venue chase win %, average 1st innings total, venue pace/spin bias, head-to-head records, dew factor indicators.
6. **Phase Indicators (6)**: Powerplay (1–6 ov), Middle (7–15 ov), Death (16–20 ov) flags, powerplay containment ratio, death acceleration rates.

---

## 💻 Interactive Web Application & Simulator

PRISM includes an interactive web dashboard and live match state simulator:

- **Live Match Simulator**: Interactively configure batting teams, bowling teams, venues, targets, current overs, and wickets lost.
- **Live Pipeline Audit**: Step-by-step interactive inspection of Detection, Diagnosis, Retrieval Correction, Verification, and Decision.
- **Comparative Visualizations**: Interactive Chart.js charts for model accuracy, Brier scores, and coverage-accuracy tradeoff curves.
- **Zero External Runtime Dependencies**: Standalone HTML/CSS/JS frontend powered by Tailwind CSS CDN and Chart.js.

---

## 🚀 Quickstart & Setup

### Option 1: One-Click Local Server (Zero Pip Dependencies)

Run the zero-dependency Python standalone server directly using the Python standard library:

```bash
# On Windows:
.\start_server.bat

# Or using Python directly:
python backend/server.py 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

### Option 2: Production Flask API

To run the full Flask backend with virtual environment:

```bash
# 1. Navigate to repository root
cd prism_v1

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Start Flask server
python backend/app.py
```

---

## 🌐 Deployment Options

### 1. GitHub Pages (Active & Configured)
The repository includes a ready-to-run GitHub Actions deployment workflow at `.github/workflows/deploy.yml` and static assets in both `frontend/` and `docs/`:
1. Push your repository to GitHub.
2. In your GitHub repository, navigate to **Settings > Pages**.
3. Under **Build and deployment > Source**, select **GitHub Actions** (or select **Deploy from a branch > main > /docs**).
4. The website will automatically deploy to `https://<username>.github.io/<repo-name>/`.

### 2. Netlify Deployment
Preconfigured with `netlify.toml`:
- Simply connect your GitHub repository to Netlify.
- Publish directory is set to `frontend/`.
- Instant live URL with global CDN.

### 3. Vercel Deployment
Preconfigured with `vercel.json`:
- Import the repository on Vercel.
- Deploys out of the box with zero additional configuration.

### 4. Cloud Backend (Render / Railway / Heroku)
The backend includes `Procfile` (`web: gunicorn app:app`) and `runtime.txt` (`python-3.11.9`) for instant deployment on PaaS providers like Render, Railway, or Heroku.

---

## 🔌 REST API Reference

The backend exposes full programmatic access to PRISM's pipeline:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Health check and server runtime status. |
| `/api/pipeline/overview` | `GET` | High-level summary of the 5 PRISM stages and architectural principles. |
| `/api/pipeline/stages` | `GET` | Detailed metadata for each stage (algorithms, inputs, outputs). |
| `/api/stats` | `GET` | Empirical benchmark statistics, accuracy metrics, and sample counts. |
| `/api/models` | `GET` | Performance benchmarks of baseline models (LogReg, RF, GBM). |
| `/api/features` | `GET` | Complete feature dictionary spanning all 50+ engineered predictors. |
| `/api/evaluation/metrics` | `GET` | Comprehensive evaluation data including coverage-accuracy tradeoffs. |
| `/api/robustness` | `GET` | Stress-testing results under noise, dropout, and temporal shifts. |
| `/api/metadata/teams` | `GET` | List of IPL franchises, colors, and titles. |
| `/api/metadata/venues` | `GET` | List of IPL stadiums, average scores, and chase biases. |
| `/api/predictions/simulate` | `POST` | Execute end-to-end PRISM pipeline on a custom match state payload. |

### Example Simulation Request (`POST /api/predictions/simulate`)

```json
{
  "batting_first": "CSK",
  "chasing": "MI",
  "venue": "Wankhede Stadium, Mumbai",
  "target": 185,
  "overs_completed": 14.0,
  "current_score": 118,
  "wickets_lost": 4
}
```

### Response:
```json
{
  "input_state": { "required_run_rate": 11.17, "current_run_rate": 8.43 },
  "pipeline": {
    "stage1_detection": { "chasing_win_probability": 0.42, "confidence": 0.68 },
    "stage2_diagnosis": { "risk_score": 0.65, "risk_level": "HIGH" },
    "stage3_correction": { "proposed_correction": 0.51, "delta": 0.09 },
    "stage4_verification": { "harm_reduction_passed": true, "consensus_score": 0.84 },
    "stage5_decision": {
      "action": "CORRECT",
      "final_winner": "MI",
      "final_chasing_probability": 0.51,
      "rationale": "High error risk verified. Historical manifold consensus confirms correction."
    }
  }
}
```

---

## 📂 Project Structure

```
prism_v1/
├── .github/
│   └── workflows/
│       └── deploy.yml              # Automated GitHub Pages CI/CD workflow
├── backend/
│   ├── app.py                      # Flask REST API server (CORS, Gunicorn-ready)
│   ├── server.py                   # Zero-dependency Python standard library server
│   ├── requirements.txt            # Python dependencies for backend API
│   ├── Procfile                    # Deployment entry point for PaaS platforms
│   └── runtime.txt                 # Python runtime specification (3.11.9)
├── frontend/
│   └── index.html                  # Interactive dashboard & match simulator (SPA)
├── docs/
│   ├── index.html                  # GitHub Pages mirror for instant hosting
│   ├── architecture.md             # System architecture and data flow
│   ├── experiment_protocol.md      # Scientific evaluation protocol
│   ├── feature_dictionary.md       # 50+ feature definitions and schemas
│   └── ...                         # Detailed scientific reports
├── configs/
│   ├── baseline.yaml               # Baseline model hyperparameters
│   ├── detection.yaml              # Risk detector configuration & thresholds
│   ├── correction.yaml             # Retrieval index & KNN weights
│   ├── verification.yaml           # Multi-evidence harm reduction thresholds
│   └── pipeline.yaml               # Orchestrator parameters
├── data/
│   ├── raw/                        # Raw Cricsheet match files (YAML/JSON)
│   └── processed/                  # Cleaned deliveries & state snapshots (Parquet)
├── src/
│   ├── interfaces.py               # Abstract base classes and data contracts
│   ├── prism.py                    # Main pipeline orchestrator
│   ├── baseline/                   # Model training, inference, and calibration
│   ├── signals/                    # Entropy, confidence, and reliability extraction
│   ├── detection/                  # Meta-learning error risk detector
│   ├── correction/                 # KNN retrieval & dynamic candidate selection
│   ├── verification/               # Multi-evidence verification engine
│   ├── evaluation/                 # Metrics, robustness tests, and visualizations
│   └── data/                       # Ingestion, feature engineering, and splitting
├── scripts/
│   ├── run_prism.py                # Pipeline execution script
│   └── run_robustness.py           # Robustness stress tests runner
├── start_server.bat                # Windows 1-click server launch script
├── netlify.toml                    # Netlify deployment configuration
├── vercel.json                     # Vercel deployment configuration
├── requirements.txt                # ML research requirements
└── README.md                       # Comprehensive project documentation
```

---

## 🔬 Scientific Methodology & Reproducibility

1. **Leakage-Free Match Splits**: All snapshot states belonging to any single cricket match are strictly assigned to either the Train, Validation, or Test split. Intra-match states never leak across splits.
2. **Temporal Validity**: Evaluation splits preserve historical chronology (earlier seasons in training, later seasons in testing) to reflect real deployment conditions.
3. **Out-of-Fold Reliability Signals**: Risk detectors are trained strictly on cross-validated out-of-fold predictions to prevent overfitting on known baseline residuals.
4. **Principled Abstention**: Selective accuracy is evaluated as a function of coverage, providing clear Pareto tradeoff curves for risk-sensitive deployments.

---

## 📜 Citation & License

This project is licensed under the MIT License.

```bibtex
@article{Swarup2026prism,
  title={PRISM: A Post-Prediction Reliability Framework for Evidence-Guided Correction and Selective Abstention},
  author={Swarup, Omm and Hiremath, Shrishail M.},
  year={2026},
  eprint={8345564},
  archivePrefix={arXiv},
  primaryClass={cs.NI},
}
```
