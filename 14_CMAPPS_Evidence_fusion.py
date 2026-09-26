# ============================================================
# AERIS Project — CMAPSS Fusion Evidence Generator
# Converts CMAPSS evidence objects into Dempster-Shafer ready
# belief/uncertainty representation.
# ============================================================

import json
import glob
import os

CMAPSS_EVIDENCE_DIR = "evidence_objects"
OUTPUT_DIR = "Fusion_Evidence/CMAPSS"

HIGH_RISK_RUL = 30
MEDIUM_RISK_RUL = 80

def determine_hypothesis(rul):
    if rul <= HIGH_RISK_RUL:
        return "High Risk", f"Predicted RUL ({rul:.1f} cycles) is at or below the high-risk threshold ({HIGH_RISK_RUL})"
    elif rul <= MEDIUM_RISK_RUL:
        return "Medium Risk", f"Predicted RUL ({rul:.1f} cycles) falls within the medium-risk band ({HIGH_RISK_RUL}-{MEDIUM_RISK_RUL})"
    else:
        return "Low Risk", f"Predicted RUL ({rul:.1f} cycles) exceeds the medium-risk threshold ({MEDIUM_RISK_RUL})"

def compute_evidence_quality(evidence):
    """
    Combines the model's own prediction confidence with SHAP explanation
    availability and data completeness, rather than trusting raw
    confidence alone.

    NOTE (limitation to state in your report): 'confidence' here is
    computed as 1 - |predicted_RUL - actual_RUL| / 125, which requires
    KNOWN ground-truth RUL. This works for validating the pipeline on
    your labeled test set, but a real deployment would need a genuine
    model-intrinsic uncertainty estimate instead (e.g. MC dropout,
    prediction intervals, ensembling) since true RUL is unknown at
    inference time in production.
    """
    prediction_confidence = evidence.get("confidence", 0.5)
    shap_available = 1.0 if evidence.get("shap_top_features") else 0.5
    data_completeness = 1.0  # CMAPSS sensor data is fully populated after preprocessing

    WEIGHTS = {"prediction_confidence": 0.65, "shap_available": 0.15, "data_completeness": 0.20}

    overall = (
        prediction_confidence * WEIGHTS["prediction_confidence"] +
        shap_available * WEIGHTS["shap_available"] +
        data_completeness * WEIGHTS["data_completeness"]
    )
    return round(overall, 3), {
        "prediction_confidence": round(prediction_confidence, 3),
        "shap_available": round(shap_available, 3),
        "data_completeness": round(data_completeness, 3),
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)
cmapss_files = glob.glob(f"{CMAPSS_EVIDENCE_DIR}/cmapss_evidence_*.json")
print(f"Found {len(cmapss_files)} CMAPSS evidence object(s) to convert.")

for filepath in cmapss_files:
    with open(filepath, "r") as f:
        cmapss_evidence = json.load(f)

    rul = cmapss_evidence["prediction"]["RUL"]
    hypothesis, hypothesis_reason = determine_hypothesis(rul)
    belief, quality_components = compute_evidence_quality(cmapss_evidence)
    uncertainty = round(1 - belief, 3)

    fusion_evidence = {
        "source": "CMAPSS",
        "engine_id": cmapss_evidence.get("engine_id"),
        "timestamp_window": cmapss_evidence.get("timestamp_window"),
        "hypothesis": hypothesis,
        "hypothesis_reason": hypothesis_reason,
        "belief": belief,
        "uncertainty": uncertainty,
        "evidence_quality_components": quality_components,
        "predicted_rul": round(rul, 2),
        "probable_cause_subsystem": cmapss_evidence.get("probable_cause_subsystem"),
        "shap_top_features": cmapss_evidence.get("shap_top_features", [])
    }

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(OUTPUT_DIR, f"cmapss_fusion_{base_name}.json")
    with open(output_path, "w") as f:
        json.dump(fusion_evidence, f, indent=2)

    print(f"\nProcessed: {base_name}")
    print(f"  Hypothesis: {hypothesis}  ({hypothesis_reason})")
    print(f"  Belief: {belief}")
    print(f"  Saved to: {output_path}")

print(f"\nDone. {len(cmapss_files)} fusion evidence object(s) generated in '{OUTPUT_DIR}/'.")