# ============================================================
# AERIS Project — Phase 1 Explainable Investigation Output Generator
# Consolidates the Dempster-Shafer result with full supporting evidence
# from all three sources into the final investigation report.
# ============================================================

import json
import glob
import os
from datetime import datetime, timezone

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
DS_RESULT_PATH = "DempsterShafer/Unified_Aircraft_Risk_Assessment.json"
CMAPSS_EVIDENCE_DIR = "evidence_objects"
NTSB_EVIDENCE_DIR = "Evidence_Objects/NTSB"
NGAFID_EVIDENCE_DIR = "AERIS_NGAFID_final_deliverables"
OUTPUT_DIR = "Reports"

# ------------------------------------------------------------
# 1. LOAD THE UNIFIED DS RESULT
# ------------------------------------------------------------
with open(DS_RESULT_PATH, "r") as f:
    ds_result = json.load(f)

print("=" * 70)
print("GENERATING PHASE 1 EXPLAINABLE INVESTIGATION OUTPUT")
print("=" * 70)

# ------------------------------------------------------------
# 2. LOAD THE FULL DETAILED EVIDENCE FROM EACH SOURCE
#    (not just the fusion summary -- the rich original evidence objects)
# ------------------------------------------------------------
cmapss_file = glob.glob(f"{CMAPSS_EVIDENCE_DIR}/cmapss_evidence_*.json")[0]
with open(cmapss_file, "r") as f:
    cmapss_detail = json.load(f)

ntsb_file = glob.glob(f"{NTSB_EVIDENCE_DIR}/ntsb_evidence_*.json")[0]
with open(ntsb_file, "r") as f:
    ntsb_detail = json.load(f)

ngafid_file = glob.glob(f"{NGAFID_EVIDENCE_DIR}/aeris_fault_evidence_object*.json")[0]
with open(ngafid_file, "r") as f:
    ngafid_detail = json.load(f)

# ------------------------------------------------------------
# 3. DETERMINE MOST PROBABLE FAULT SUBSYSTEM
# ------------------------------------------------------------
# Since CMAPSS and NTSB agree on "High-Pressure Compressor" (modulo the
# hyphen/space formatting difference we identified earlier), and NGAFID
# independently points to a different subsystem on a different case,
# we report the DOMINANT source's subsystem (the one driving the final
# risk decision) while still surfacing NGAFID's finding as supporting
# evidence -- not silently dropped, just correctly weighted.

dominant_source = max(
    ds_result["source_summary"].items(),
    key=lambda kv: kv[1]["belief"]
)
dominant_source_name = dominant_source[0]

subsystem_map = {
    "CMAPSS": cmapss_detail.get("probable_cause_subsystem"),
    "NTSB": ntsb_detail["engineering_guidance"]["recommended_subsystem"],
    "NGAFID": ngafid_detail["differential_diagnosis"]["primary_subsystem_diagnosis"],
}
most_probable_subsystem = subsystem_map[dominant_source_name]

# ------------------------------------------------------------
# 4. BUILD THE FINAL EXPLAINABLE INVESTIGATION OUTPUT
# ------------------------------------------------------------
report = {
    "aeris_output_type": "PHASE_1_EXPLAINABLE_INVESTIGATION_OUTPUT",
    "report_generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),

    "aircraft_risk_level": ds_result["final_decision"]["risk_level"],

    "overall_confidence_belief": {
        "pignistic_probability": ds_result["final_decision"]["pignistic_probability"],
        "belief_lower_bound": ds_result["hypothesis_assessment"][ds_result["final_decision"]["risk_level"]]["belief"],
        "plausibility_upper_bound": ds_result["hypothesis_assessment"][ds_result["final_decision"]["risk_level"]]["plausibility"],
        "interpretation": (
            f"The system is at least {ds_result['hypothesis_assessment'][ds_result['final_decision']['risk_level']]['belief']*100:.1f}% "
            f"certain, and at most {ds_result['hypothesis_assessment'][ds_result['final_decision']['risk_level']]['plausibility']*100:.1f}% "
            f"certain, that the aircraft falls into the '{ds_result['final_decision']['risk_level']}' category."
        )
    },

    "most_probable_fault_subsystem": {
        "subsystem": most_probable_subsystem,
        "primary_evidence_source": dominant_source_name,
        "note": (
            "This subsystem is drawn from the highest-belief source. Other "
            "sources may point to different subsystems if they describe a "
            "different case (see individual source evidence below)."
        )
    },

    "supporting_sensor_evidence_cmapss": {
        "predicted_rul_cycles": cmapss_detail["prediction"]["RUL"],
        "prediction_confidence": cmapss_detail["confidence"],
        "top_contributing_sensors_shap": cmapss_detail["shap_top_features"],
        "inferred_subsystem": cmapss_detail.get("probable_cause_subsystem")
    },

    "supporting_historical_investigation_ntsb": {
        "matched_report_id": ntsb_detail["metadata"]["event_id"],
        "aircraft_involved": ntsb_detail["investigation_summary"]["aircraft"],
        "semantic_similarity": ntsb_detail["retrieval_assessment"]["semantic_similarity"],
        "probable_cause": ntsb_detail["causal_evidence"]["probable_cause"],
        "evidence_confidence": ntsb_detail["evidence_quality"]["overall_evidence_confidence"]
    },

    "supporting_flight_evidence_ngafid": {
        "flight_master_index": ngafid_detail["metadata"]["flight_master_index"],
        "failure_risk_probability": ngafid_detail["screening_assessment"]["failure_risk_probability"],
        "diagnosed_subsystem": ngafid_detail["differential_diagnosis"]["primary_subsystem_diagnosis"],
        "diagnosis_confidence": ngafid_detail["differential_diagnosis"]["diagnosis_confidence"],
        "top_anomaly_sensors": [
            d["sensor_feature"] for d in ngafid_detail["screening_assessment"]["top_anomaly_sensor_drivers"][:3]
        ]
    },

    "fusion_transparency": {
        "sources_combined": ds_result["sources_combined"],
        "source_level_hypotheses": ds_result["source_summary"],
        "inter_source_conflict": ds_result["pairwise_conflicts"],
        "note": (
            "High conflict values indicate sources described different "
            "underlying cases (a known limitation when combining independent "
            "benchmark datasets in this Phase 1 prototype -- see project "
            "documentation for details)."
        )
    }
}

# ------------------------------------------------------------
# 5. SAVE (JSON + human-readable text version)
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

json_path = os.path.join(OUTPUT_DIR, "Phase1_Explainable_Investigation_Output.json")
with open(json_path, "w") as f:
    json.dump(report, f, indent=2)

# Human-readable version for the report/appendix
text_path = os.path.join(OUTPUT_DIR, "Phase1_Explainable_Investigation_Output.txt")
with open(text_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("AERIS PHASE 1 -- EXPLAINABLE INVESTIGATION OUTPUT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated: {report['report_generated']}\n\n")
    f.write(f"AIRCRAFT RISK LEVEL: {report['aircraft_risk_level']}\n")
    f.write(f"  {report['overall_confidence_belief']['interpretation']}\n\n")
    f.write(f"MOST PROBABLE FAULT SUBSYSTEM: {report['most_probable_fault_subsystem']['subsystem']}\n")
    f.write(f"  (primary evidence source: {report['most_probable_fault_subsystem']['primary_evidence_source']})\n\n")
    f.write("SUPPORTING SENSOR EVIDENCE (CMAPSS):\n")
    f.write(f"  Predicted RUL: {report['supporting_sensor_evidence_cmapss']['predicted_rul_cycles']} cycles\n")
    f.write(f"  Top SHAP features: {', '.join(report['supporting_sensor_evidence_cmapss']['top_contributing_sensors_shap'])}\n\n")
    f.write("SUPPORTING HISTORICAL INVESTIGATION (NTSB):\n")
    f.write(f"  Matched report: {report['supporting_historical_investigation_ntsb']['matched_report_id']}\n")
    f.write(f"  Similarity: {report['supporting_historical_investigation_ntsb']['semantic_similarity']}\n")
    f.write(f"  Probable cause: {report['supporting_historical_investigation_ntsb']['probable_cause'][:200]}...\n\n")
    f.write("SUPPORTING FLIGHT EVIDENCE (NGAFID):\n")
    f.write(f"  Failure risk probability: {report['supporting_flight_evidence_ngafid']['failure_risk_probability']}\n")
    f.write(f"  Diagnosed subsystem: {report['supporting_flight_evidence_ngafid']['diagnosed_subsystem']}\n\n")
    f.write("FUSION TRANSPARENCY:\n")
    f.write(f"  Sources combined: {', '.join(report['fusion_transparency']['sources_combined'])}\n")
    f.write(f"  Inter-source conflict: {report['fusion_transparency']['inter_source_conflict']}\n")

print(f"\nSaved JSON report to: {json_path}")
print(f"Saved readable report to: {text_path}")
print("\n" + "=" * 70)
print("FINAL REPORT PREVIEW")
print("=" * 70)
print(json.dumps(report, indent=2))