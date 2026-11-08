"""
Unit Tests for PRISM Pipeline & Verification Engine
Run with: python -m unittest discover tests
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prism import PRISMPipeline

class TestPRISMPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = PRISMPipeline()
        self.pipeline.initialize()

    def test_pipeline_initialization(self):
        self.assertTrue(self.pipeline.is_ready)

    def test_low_risk_accept_decision(self):
        # A comfortable chase scenario: 120 target, 110 runs in 12 overs with only 1 wicket lost
        state = {
            "over_completed": 12.0,
            "current_score": 110,
            "wickets_lost": 1,
            "target": 120,
            "venue": "Wankhede Stadium",
            "batting_team": "Mumbai Indians",
            "bowling_team": "Chennai Super Kings"
        }
        res = self.pipeline.predict_state(state)
        self.assertIn("decision", res)
        self.assertIn("error_risk_score", res)
        self.assertIn("baseline_probability", res)
        # In a comfortable chase, win probability should be high
        self.assertGreater(res["baseline_probability"], 0.70)
        self.assertLessEqual(res["error_risk_score"], 0.50)
        self.assertEqual(res["decision"], "ACCEPT")

    def test_high_pressure_correction_or_abstain(self):
        # Extreme pressure: 200 target, 75 runs in 15 overs with 7 wickets lost
        state = {
            "over_completed": 15.0,
            "current_score": 75,
            "wickets_lost": 7,
            "target": 200,
            "venue": "Eden Gardens",
            "batting_team": "Kolkata Knight Riders",
            "bowling_team": "Sunrisers Hyderabad"
        }
        res = self.pipeline.predict_state(state)
        self.assertIn(res["decision"], ["ACCEPT", "CORRECT", "ABSTAIN"])
        self.assertLess(res["baseline_probability"], 0.30)

    def test_ambiguous_state_entropy(self):
        # Evenly poised match: 180 target, 90 runs at 10 overs, 4 wickets down
        state = {
            "over_completed": 10.0,
            "current_score": 90,
            "wickets_lost": 4,
            "target": 180,
            "venue": "M. Chinnaswamy Stadium",
            "batting_team": "Royal Challengers Bengaluru",
            "bowling_team": "Rajasthan Royals"
        }
        res = self.pipeline.predict_state(state)
        # Entropy should be high near 0.5 probability
        self.assertGreater(res["entropy"], 0.80)

if __name__ == "__main__":
    unittest.main()
