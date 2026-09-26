# ============================================================
# AERIS Project — NGAFID Fusion Evidence Generator
# Converts the FAULT_EVIDENCE_OBJECT into a Dempster-Shafer ready
# belief/uncertainty representation.
# ============================================================

import json
import glob
import os

# ------------------------------------------------------------
# 0. CONFIG — paths matched to your actual folder structure
#    (AERIS/AERIS_NGAFID_final_deliverables/aeris_fault_evidence_object.json)
# ------------------------------------------------------------
NGAFID_EVIDENCE_DIR = "AERIS_NGAFID_final_deliverables"
OUTPUT_DIR = "Fusion_Evidence/NGAFID"

# Risk bands anchored to the model's own screening_decision_threshold,
# rather than an arbitrary fixed number -- keeps bands consistent even
# if the screening model is retrained with a different threshold later.
# NOTE (state in report): this multiplier (1.4) is a reasoned but
# manually chosen design decision, not derived from a formal calibration
# study -- worth flagging as a parameter for future tuning.
HIGH_RISK_MULTIPLIER = 1.4


# ------------------------------------------------------------
# 1. RISK HYPOTHESIS DETERMINATION
# ------------------------------------------------------------
def determine_hypothesis(risk_prob, threshold):
    high_cutoff = min(threshold * HIGH_RISK_MULTIPLIER, 1.0)

    if risk_prob >= high_cutoff:
        return "High Risk", (
            f"Failure risk probability ({risk_prob:.3f}) substantially exceeds "
            f"the screening threshold ({threshold})"
        )
    elif risk_prob >= threshold:
        return "Medium Risk", (
            f"Failure risk probability ({risk_prob:.3f}) exceeds the screening "
            f"threshold ({threshold}) but not by a wide margin"
        )
    else:
        return "Low Risk", (
            f"Failure risk probability ({risk_prob:.3f}) is below the screening "
            f"threshold ({threshold})"
        )


# ------------------------------------------------------------
# 2. MULTI-FACTOR EVIDENCE QUALITY
# ------------------------------------------------------------
def compute_evidence_quality(evidence):
    """
    Combines the screening model's failure risk probability with the
    differential-diagnosis confidence (how sure the system is about WHICH
    subsystem is at fault, not just THAT something is wrong), plus how
    consistently the top SHAP-driven sensors agree on elevating risk.

    A high failure_risk_probability with a LOW diagnosis_confidence means
    "something is likely wrong, but we're not sure what" -- meaningfully
    weaker evidence than when both agree, so both are folded into the
    overall belief rather than using risk probability alone.
    """
    screening = evidence["screening_assessment"]
    diagnosis = evidence["differential_diagnosis"]

    failure_risk = screening["failure_risk_probability"]
    diagnosis_confidence = diagnosis["diagnosis_confidence"]

    drivers = screening.get("top_anomaly_sensor_drivers", [])
    elevating = sum(1 for d in drivers if d.get("impact_direction") == "RISK ELEVATING")
    driver_agreement = elevating / len(drivers) if drivers else 0.5

    WEIGHTS = {
        "failure_risk": 0.45,
        "diagnosis_confidence": 0.35,
        "driver_agreement": 0.20,
    }

    overall = (
        failure_risk * WEIGHTS["failure_risk"] +
        diagnosis_confidence * WEIGHTS["diagnosis_confidence"] +
        driver_agreement * WEIGHTS["driver_agreement"]
    )

    return round(overall, 3), {
        "failure_risk_probability": round(failure_risk, 3),
        "diagnosis_confidence": round(diagnosis_confidence, 3),
        "driver_agreement": round(driver_agreement, 3),
    }


# ------------------------------------------------------------
# 3. MAIN CONVERSION
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only match the actual evidence object file(s) -- excludes
# aeris_evaluation_metrics.json, feature_names.json, label_mapping.json
# which live in the same folder but are not evidence objects.
ngafid_files = glob.glob(f"{NGAFID_EVIDENCE_DIR}/aeris_fault_evidence_object*.json")

print("=" * 70)
print("NGAFID FUSION EVIDENCE GENERATION")
print("=" * 70)
print(f"Looking in: {NGAFID_EVIDENCE_DIR}")
print(f"Found {len(ngafid_files)} NGAFID evidence object(s) to convert.")
print(f"Files matched: {ngafid_files}")

if len(ngafid_files) == 0:
    raise FileNotFoundError(
        f"No files matched 'aeris_fault_evidence_object*.json' inside "
        f"'{NGAFID_EVIDENCE_DIR}'. Check the folder path and filename."
    )

for filepath in ngafid_files:
    with open(filepath, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    screening = evidence["screening_assessment"]
    diagnosis = evidence["differential_diagnosis"]

    risk_prob = screening["failure_risk_probability"]
    threshold = screening["screening_decision_threshold"]

    hypothesis, hypothesis_reason = determine_hypothesis(risk_prob, threshold)
    belief, quality_components = compute_evidence_quality(evidence)
    uncertainty = round(1 - belief, 3)

    fusion_evidence = {
        "source": "NGAFID",
        "flight_master_index": evidence["metadata"]["flight_master_index"],
        "hypothesis": hypothesis,
        "hypothesis_reason": hypothesis_reason,
        "belief": belief,
        "uncertainty": uncertainty,
        "evidence_quality_components": quality_components,
        "primary_subsystem_diagnosis": diagnosis["primary_subsystem_diagnosis"],
        "top3_differential_candidates": diagnosis["top3_differential_candidates"],
        "top_anomaly_sensor_drivers": [
            d["sensor_feature"] for d in screening.get("top_anomaly_sensor_drivers", [])
        ]
    }

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(OUTPUT_DIR, f"ngafid_fusion_{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fusion_evidence, f, indent=2)

    print(f"\nProcessed: {base_name}")
    print(f"  Flight index: {fusion_evidence['flight_master_index']}")
    print(f"  Hypothesis  : {hypothesis}  ({hypothesis_reason})")
    print(f"  Belief      : {belief}   Uncertainty: {uncertainty}")
    print(f"  Primary subsystem: {diagnosis['primary_subsystem_diagnosis']}")
    print(f"  Saved to    : {output_path}")

print(f"\nDone. {len(ngafid_files)} fusion evidence object(s) generated in '{OUTPUT_DIR}/'.")

print("\n" + "=" * 70)
print("EXAMPLE FUSION EVIDENCE")
print("=" * 70)
print(json.dumps(fusion_evidence, indent=2))