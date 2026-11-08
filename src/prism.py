"""
PRISM: Prediction Reliability and Intelligent Self-Correction Mechanism
Main Pipeline Orchestrator

Flow:
    Match State -> 1. Detect (Baseline Model)
               -> 2. Diagnose (Reliability Signals & Error Detector)
               -> 3. Correct (Historical State Retrieval & Dynamic Selector)
               -> 4. Verify (Harm Reduction & Statistical Checks)
               -> 5. Decision (ACCEPT, CORRECT, ABSTAIN)
"""

from pathlib import Path
import math
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to load heavy scientific libraries if installed in environment
try:
    import numpy as np
    import pandas as pd
    import joblib
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

class PRISMPipeline:
    """
    End-to-end PRISM framework orchestrator.
    Supports both trained joblib model inference (when scikit-learn is installed)
    and zero-dependency mathematical heuristic inference.
    """

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        self.models = None
        self.similarity_reference = None
        self.is_ready = False
        self.use_trained_models = HAS_ML_DEPS

    def initialize(self, train_data_path=None):
        """
        Load frozen models and build historical similarity reference.
        """
        if self.use_trained_models:
            try:
                from src.evaluation.final_evaluation import (
                    load_models,
                    build_similarity_reference
                )
                self.models = load_models()
                if train_data_path is None:
                    train_data_path = PROJECT_ROOT / "data" / "match_states.parquet"

                if Path(train_data_path).exists():
                    data = pd.read_parquet(train_data_path)
                    if "season" in data.columns:
                        train_data = data[data["season"].isin(["2020/21", "2021", "2022", "2023"])].copy()
                        if len(train_data) == 0:
                            train_data = data.copy()
                    else:
                        train_data = data.copy()
                    self.similarity_reference = build_similarity_reference(train_data)
                self.is_ready = True
                return self
            except Exception as e:
                print(f"[PRISM] Notice: Fallback to standalone mode ({e})")
                self.use_trained_models = False

        self.is_ready = True
        return self

    def predict_state(self, state_dict):
        """
        Run PRISM on a single match state dictionary.
        """
        if not self.is_ready:
            self.initialize()

        if self.use_trained_models and self.models is not None:
            return self._predict_with_models(state_dict)
        else:
            return self._predict_heuristic(state_dict)

    def _predict_with_models(self, state_dict):
        from src.evaluation.final_evaluation import (
            MODEL_FEATURES,
            calculate_reliability_signals,
            generate_error_risk,
            generate_similarity_evidence,
            apply_prism_policy
        )
        df = pd.DataFrame([state_dict])
        if "runs_required" not in df.columns:
            df["runs_required"] = max(0, df["target"].iloc[0] - df["current_score"].iloc[0])
        if "balls_remaining" not in df.columns:
            df["balls_remaining"] = max(1, (20.0 - df["over_completed"].iloc[0]) * 6)
        if "current_run_rate" not in df.columns:
            df["current_run_rate"] = df["current_score"].iloc[0] / max(0.5, df["over_completed"].iloc[0])
        if "required_run_rate" not in df.columns:
            df["required_run_rate"] = df["runs_required"].iloc[0] / (df["balls_remaining"].iloc[0] / 6.0)
        if "runs_last_3_overs" not in df.columns:
            df["runs_last_3_overs"] = df["current_run_rate"].iloc[0] * 3
        if "wickets_last_3_overs" not in df.columns:
            df["wickets_last_3_overs"] = 1

        base_model = self.models["random_forest"]
        baseline_prob = float(base_model.predict_proba(df[MODEL_FEATURES])[:, 1][0])
        baseline_pred = int(baseline_prob >= 0.5)

        signals = calculate_reliability_signals([baseline_prob])
        confidence = float(signals["confidence"].iloc[0])
        entropy = float(signals["predictive_entropy"].iloc[0])

        error_risk = float(generate_error_risk(signals, self.models["error_detector"])[0])

        if self.similarity_reference is not None:
            sim_evidence = generate_similarity_evidence(df, self.similarity_reference)
            sim_prob = float(sim_evidence["similarity_probability"].iloc[0])
            sim_pred = int(sim_evidence["similarity_prediction"].iloc[0])
            sim_conf = float(sim_evidence["similarity_confidence"].iloc[0])
            support_win = int(sim_evidence["support_chasing_win"].iloc[0])
            support_loss = int(sim_evidence["support_chasing_loss"].iloc[0])
            neighbor_count = int(sim_evidence["neighbor_count"].iloc[0])
        else:
            sim_prob, sim_pred, sim_conf = baseline_prob, baseline_pred, confidence
            support_win, support_loss, neighbor_count = 25, 25, 50

        eval_row = df.copy()
        eval_row["baseline_prediction"] = baseline_pred
        eval_row["baseline_probability"] = baseline_prob
        eval_row["error_risk_score"] = error_risk
        eval_row["similarity_prediction"] = sim_pred
        eval_row["similarity_probability"] = sim_prob
        eval_row["similarity_confidence"] = sim_conf
        eval_row["support_chasing_win"] = support_win
        eval_row["support_chasing_loss"] = support_loss
        eval_row["neighbor_count"] = neighbor_count

        policy_result = apply_prism_policy(eval_row)
        decision = policy_result["decision"].iloc[0]
        final_prediction = policy_result["final_prediction"].iloc[0]
        reason = policy_result["decision_reason"].iloc[0]

        return {
            "mode": "Trained Models (Scikit-Learn)",
            "baseline_probability": round(baseline_prob, 4),
            "baseline_prediction": baseline_pred,
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "error_risk_score": round(error_risk, 4),
            "similarity_probability": round(sim_prob, 4),
            "similarity_prediction": sim_pred,
            "decision": decision,
            "final_prediction": None if np.isnan(final_prediction) else int(final_prediction),
            "decision_reason": reason
        }

    def _predict_heuristic(self, state_dict):
        target = float(state_dict.get("target", 180))
        overs = float(state_dict.get("over_completed", 14.0))
        score = float(state_dict.get("current_score", 110))
        wickets = float(state_dict.get("wickets_lost", 4))

        balls_left = max(1.0, (20.0 - overs) * 6.0)
        runs_needed = max(0.0, target - score)
        rrr = runs_needed / (balls_left / 6.0)
        wickets_in_hand = 10.0 - wickets

        # Heuristic win probability
        par_rrr = 8.5
        prob = 0.50 - ((rrr - par_rrr) * 0.055) + ((wickets_in_hand - 5.0) * 0.075)
        prob = max(0.04, min(0.96, prob))

        confidence = abs(prob - 0.5) * 2.0
        entropy = -(prob * math.log2(max(1e-5, prob)) + (1 - prob) * math.log2(max(1e-5, 1 - prob)))

        is_ambiguous = 0.40 <= prob <= 0.60
        is_high_pressure = (rrr > 11.0 and wickets_in_hand < 5)
        risk = 0.30 + (0.32 if is_ambiguous else 0.04) + (0.26 if is_high_pressure else 0.02)
        risk = min(0.98, max(0.10, risk))

        sim_prob = prob + 0.08 if risk > 0.60 else prob
        sim_pred = int(sim_prob >= 0.5)

        if risk <= 0.50:
            decision = "ACCEPT"
            final_pred = int(prob >= 0.5)
            reason = "Baseline prediction below frozen error-risk threshold."
        elif risk > 0.50 and abs(sim_prob - prob) > 0.05:
            decision = "CORRECT"
            final_pred = sim_pred
            reason = "High-risk baseline prediction with strong historical similarity evidence."
        else:
            decision = "ABSTAIN"
            final_pred = None
            reason = "High-risk baseline prediction without sufficient evidence for a safe correction."

        return {
            "mode": "Standalone Engine (Zero-Dependency)",
            "baseline_probability": round(prob, 4),
            "baseline_prediction": int(prob >= 0.5),
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "error_risk_score": round(risk, 4),
            "similarity_probability": round(sim_prob, 4),
            "similarity_prediction": sim_pred,
            "decision": decision,
            "final_prediction": final_pred,
            "decision_reason": reason
        }

def main():
    print("==================================================")
    print(" PRISM: Pipeline Orchestrator Initialized")
    print("==================================================")
    pipeline = PRISMPipeline()
    pipeline.initialize()

    sample_state = {
        "over_completed": 14.0,
        "current_score": 118,
        "wickets_lost": 4,
        "target": 185,
        "venue": "Wankhede Stadium",
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
    }

    print("\nRunning PRISM on sample match state:")
    for k, v in sample_state.items():
        print(f"  {k}: {v}")

    result = pipeline.predict_state(sample_state)
    print("\nPRISM Pipeline Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("\n[SUCCESS] Pipeline execution finished.")

if __name__ == '__main__':
    main()
