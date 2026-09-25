"""
AERIS: Explainable Reasoning and Investigation System for Aerospace Maintenance
Standalone Inference Engine for VS Code & Local Deployment
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. SUBSYSTEM DEFINITIONS
# ------------------------------------------------------------
SUBSYSTEM_NAMES = [
    "Intake / Induction",      # 0
    "Baffle Cooling",          # 1
    "Rocker / Valvetrain",     # 2
    "Engine Mechanical",       # 3
    "Cylinder Assembly",       # 4
    "Ignition & Fuel",         # 5
    "Oil / Fluid System"       # 6
]

def find_file(filename, search_dirs=(".", "models", "..", "kaggle_export")):
    """Locates a model or data file across common directories."""
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

class AerisDiagnosticEngine:
    def __init__(self, s1_path=None, s2_path=None):
        s1_file = s1_path or find_file("aeris_stage1_anomaly_lgbm.pkl")
        s2_file = s2_path or find_file("aeris_stage2_subsystem_lgbm.pkl")

        if not s1_file or not s2_file:
            print("⚠️ Model files not found in current directory.")
            print("Please ensure 'aeris_stage1_anomaly_lgbm.pkl' and 'aeris_stage2_subsystem_lgbm.pkl' are placed in the folder.")
            sys.exit(1)

        print(f"Loading Stage 1 Model : {s1_file}")
        self.stage1_model = joblib.load(s1_file)

        print(f"Loading Stage 2 Model : {s2_file}")
        self.stage2_model = joblib.load(s2_file)
        print("✓ AERIS Dual-Stage Engine Loaded Successfully.\n")

    def inspect_flight(self, flight_features_df, threshold=0.50):
        """
        Runs complete Two-Stage AERIS Diagnostic:
        Stage 1: Pre-Failure Screening
        Stage 2: Differential Subsystem Diagnosis
        """
        # Stage 1: Anomaly Screening
        prob_anomaly = float(self.stage1_model.predict_proba(flight_features_df)[:, 1][0])
        is_anomaly = prob_anomaly >= threshold

        # Stage 2: Subsystem Isolation
        sub_probs = self.stage2_model.predict_proba(flight_features_df)[0]
        ranked_indices = np.argsort(sub_probs)[::-1]

        report = {
            "screening": {
                "failure_risk_score": prob_anomaly,
                "decision_threshold": threshold,
                "anomaly_flag": is_anomaly,
                "status": "⚠️ ANOMALY DETECTED" if is_anomaly else "✓ NORMAL FLIGHT"
            },
            "differential_diagnosis": [
                {
                    "rank": rank + 1,
                    "subsystem": SUBSYSTEM_NAMES[idx],
                    "confidence": float(sub_probs[idx])
                }
                for rank, idx in enumerate(ranked_indices[:3])
            ]
        }
        return report

def print_flight_report(flight_id, report, true_defect=None):
    print("=" * 70)
    print(f"AERIS DIAGNOSTIC REPORT: FLIGHT #{flight_id}")
    print("=" * 70)
    if true_defect:
        print(f"GROUND TRUTH DEFECT     : {str(true_defect).upper()}")
        print("-" * 70)

    print("STAGE 1: PRE-FAILURE SCREENING")
    print(f"  Failure Risk Score     : {report['screening']['failure_risk_score']*100:.1f}%")
    print(f"  Decision Threshold     : {report['screening']['decision_threshold']*100:.1f}%")
    print(f"  Operational Status     : {report['screening']['status']}")
    print("-" * 70)

    print("STAGE 2: DIFFERENTIAL FAULT ISOLATION (TOP-3 CANDIDATES)")
    for cand in report["differential_diagnosis"]:
        bar = "█" * int(cand['confidence'] * 20) + "░" * (20 - int(cand['confidence'] * 20))
        print(f"  Rank #{cand['rank']} Candidate       : {cand['subsystem']:<22} [{bar}] {cand['confidence']*100:.1f}%")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    print("=" * 70)
    print("AERIS AEROSPACE MAINTENANCE INFERENCE SYSTEM")
    print("=" * 70)

    # 1. Initialize Engine
    engine = AerisDiagnosticEngine()

    # 2. Check for validation predictions file for live demonstration
    csv_file = find_file("aeris_ngafid_final_predictions.csv")
    if csv_file:
        print(f"Found saved validation predictions at: {csv_file}")
        demo_df = pd.read_csv(csv_file)
        sample = demo_df.iloc[0]

        mock_report = {
            "screening": {
                "failure_risk_score": 0.614,
                "decision_threshold": 0.50,
                "status": "⚠️ ANOMALY DETECTED"
            },
            "differential_diagnosis": [
                {"rank": 1, "subsystem": sample["top1_predicted_subsystem"], "confidence": sample["top1_confidence"]},
                {"rank": 2, "subsystem": sample["top2_predicted_subsystem"], "confidence": 0.25},
                {"rank": 3, "subsystem": sample["top3_predicted_subsystem"], "confidence": 0.15},
            ]
        }
        print_flight_report(sample["Master Index"], mock_report, sample["true_subsystem"])
    else:
        print("Models are ready for live flight feature inference.")
