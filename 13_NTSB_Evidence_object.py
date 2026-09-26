# ============================================================
# AERIS Project — NTSB Investigation Evidence Object Generator
# (Reads retrieved_ntsb_case.json -> produces standardized evidence object)
# ============================================================

import json
import os
from datetime import datetime, timezone

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
RETRIEVED_CASE_PATH = "Semantic_Retrieval/retrieved_ntsb_case.json"
OUTPUT_DIR = "Evidence_Objects/NTSB"

# ------------------------------------------------------------
# 1. LOAD RETRIEVED CASE
# ------------------------------------------------------------
with open(RETRIEVED_CASE_PATH, "r", encoding="utf-8") as f:
    retrieved = json.load(f)

query = retrieved["retrieval_query"]
match = retrieved["top_match"]

print("=" * 70)
print("BUILDING NTSB INVESTIGATION EVIDENCE OBJECT")
print("=" * 70)
print(f"Source event: {match['event_id']}  (similarity: {match['similarity']})")

# ------------------------------------------------------------
# 2. SUBSYSTEM / MAINTENANCE-FOCUS EXTRACTION (rule-based, explainable)
# ------------------------------------------------------------
SUBSYSTEM_KEYWORDS = {
    "compressor": "High Pressure Compressor",
    "turbine": "Turbine Section",
    "fuel": "Fuel System",
    "bleed": "Bleed Air System",
    "landing gear": "Landing Gear",
    "propeller": "Propeller System",
    "bearing": "Bearing / Rotating Assembly",
    "electrical": "Electrical System",
    "hydraulic": "Hydraulic System",
    "fan": "Fan Section",
}

def extract_subsystem(text):
    text_lower = text.lower()
    for keyword, subsystem in SUBSYSTEM_KEYWORDS.items():
        if keyword in text_lower:
            return subsystem
    return "Unspecified / General Powerplant"

def extract_maintenance_focus(findings_text):
    lower = findings_text.lower()
    if "maintenance" in lower or "service" in lower or "inspection" in lower:
        return "Maintenance / Inspection-related deficiency"
    elif "fatigue" in lower or "crack" in lower or "material" in lower:
        return "Material fatigue / design-related failure"
    elif "malfunction" in lower or "failure" in lower:
        return "Component malfunction"
    else:
        return "Unclassified causal factor"

combined_text = f"{match.get('probable_cause','')} {match.get('findings','')}"
recommended_subsystem = extract_subsystem(combined_text)
maintenance_focus = extract_maintenance_focus(match.get("findings", ""))

# ------------------------------------------------------------
# 3. EVIDENCE QUALITY / CONFIDENCE COMPUTATION
# ------------------------------------------------------------
# Instead of relying on similarity alone, combine multiple measurable
# aspects of the retrieved report into one defensible confidence score.

def is_known(value):
    return bool(value) and str(value).strip().lower() not in ("unknown", "", "none", "nan")

# Component 1: Semantic similarity (already 0-1 range from FAISS)
similarity_score = float(match.get("similarity", 0.0))

# Component 2: Causal completeness -- does the report actually explain WHY?
causal_fields_present = sum([
    is_known(match.get("probable_cause")),
    is_known(match.get("findings")),
])
causal_completeness = causal_fields_present / 2.0  # 0, 0.5, or 1.0

# Component 3: Contextual completeness -- flight phase, weather, damage, purpose
context_fields = ["flight_phase", "weather", "damage", "purpose"]
context_known_count = sum(is_known(match.get(f)) for f in context_fields)
context_completeness = context_known_count / len(context_fields)

# Component 4: Report depth -- does full report text exist and have substance?
report_text_len = len(str(match.get("report_text", "")))
report_depth_score = min(report_text_len / 2000, 1.0)  # cap at 1.0 for reports >=2000 chars

# ---- Weighted combination ----
# Similarity gets the largest weight since it drives relevance,
# but causal/context/report-depth prevent a lucky-but-empty match
# from scoring as "confident" evidence.
WEIGHTS = {
    "similarity": 0.50,
    "causal_completeness": 0.25,
    "context_completeness": 0.10,
    "report_depth": 0.15,
}

overall_evidence_confidence = (
    similarity_score * WEIGHTS["similarity"] +
    causal_completeness * WEIGHTS["causal_completeness"] +
    context_completeness * WEIGHTS["context_completeness"] +
    report_depth_score * WEIGHTS["report_depth"]
)
overall_evidence_confidence = round(overall_evidence_confidence, 3)

print(f"\nEvidence Quality Breakdown:")
print(f"  Similarity              : {similarity_score:.3f}  (weight {WEIGHTS['similarity']})")
print(f"  Causal completeness     : {causal_completeness:.3f}  (weight {WEIGHTS['causal_completeness']})")
print(f"  Context completeness    : {context_completeness:.3f}  (weight {WEIGHTS['context_completeness']})")
print(f"  Report depth            : {report_depth_score:.3f}  (weight {WEIGHTS['report_depth']})")
print(f"  --> Overall Evidence Confidence: {overall_evidence_confidence}")

# ------------------------------------------------------------
# 4. BUILD THE STANDARDIZED EVIDENCE OBJECT
# ------------------------------------------------------------
evidence_object = {
    "aeris_evidence_type": "INVESTIGATION_EVIDENCE_OBJECT",

    "metadata": {
        "event_id": match.get("event_id"),
        "retrieval_timestamp": retrieved.get("retrieval_timestamp"),
        "data_source": "NTSB Aviation Investigation Reports"
    },

    "retrieval_assessment": {
        "semantic_query": query,
        "retrieval_rank": match.get("rank", 1),
        "semantic_similarity": round(similarity_score, 4)
    },

    "investigation_summary": {
        "aircraft": f"{match.get('aircraft_make', 'Unknown')} {match.get('aircraft_model', '')}".strip(),
        "flight_phase": match.get("flight_phase", "Unknown"),
        "weather": match.get("weather", "Unknown"),
        "damage": match.get("damage", "Unknown"),
        "purpose": match.get("purpose", "Unknown")
    },

    "causal_evidence": {
        "probable_cause": match.get("probable_cause", ""),
        "findings": match.get("findings", ""),
        "fatal_injuries": match.get("fatal_injuries", 0),
        "serious_injuries": match.get("serious_injuries", 0),
        "minor_injuries": match.get("minor_injuries", 0)
    },

    "engineering_guidance": {
        "recommended_subsystem": recommended_subsystem,
        "maintenance_focus": maintenance_focus,
        "supporting_report": match.get("event_id")
    },

    "evidence_quality": {
        "overall_evidence_confidence": overall_evidence_confidence,
        "components": {
            "similarity": round(similarity_score, 3),
            "causal_completeness": round(causal_completeness, 3),
            "context_completeness": round(context_completeness, 3),
            "report_depth": round(report_depth_score, 3)
        }
    }
}

# ------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f"ntsb_evidence_{match.get('event_id')}.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(evidence_object, f, indent=2)

print(f"\nSaved NTSB Investigation Evidence Object to: {output_path}")
print("\nFinal Evidence Object:")
print(json.dumps(evidence_object, indent=2))