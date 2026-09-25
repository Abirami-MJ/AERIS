# ============================================================
# AERIS Project — CMAPSS Fusion Evidence Generator
# (Converts detailed evidence objects into normalized DS-ready format)
# ============================================================

import json
import glob
import os

# ------------------------------------------------------------
# 0. CONFIG — RUL RISK BANDS (adjust thresholds as your project defines them)
# ------------------------------------------------------------
def rul_to_risk_hypothesis(rul):
    """Maps a predicted RUL value to a risk category hypothesis."""
    if rul <= 30:
        return "High Risk"
    elif rul <= 80:
        return "Medium Risk"
    else:
        return "Low Risk"

# ------------------------------------------------------------
# 1. LOAD ALL DETAILED CMAPSS EVIDENCE OBJECTS
# ------------------------------------------------------------
input_folder = "evidence_objects"          # where your detailed objects are saved
output_folder = "Fusion_Evidence/CMAPSS"   # new normalized output location

os.makedirs(output_folder, exist_ok=True)

evidence_files = glob.glob(f"{input_folder}/cmapss_evidence_*.json")
print(f"Found {len(evidence_files)} detailed CMAPSS evidence objects.")

# ------------------------------------------------------------
# 2. CONVERT EACH ONE INTO FUSION-READY FORMAT
# ------------------------------------------------------------
converted = 0
for filepath in evidence_files:
    with open(filepath, "r") as f:
        detailed = json.load(f)

    rul = detailed["prediction"]["RUL"]
    confidence = detailed["confidence"]

    hypothesis = rul_to_risk_hypothesis(rul)

    fusion_evidence = {
        "source": "CMAPSS",
        "engine_id": detailed.get("engine_id", None),
        "hypothesis": hypothesis,
        "belief": round(confidence, 3),
        "uncertainty": round(1 - confidence, 3)
    }

    filename = os.path.basename(filepath).replace("cmapss_evidence_", "cmapss_fusion_evidence_")
    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w") as f:
        json.dump(fusion_evidence, f, indent=2)

    converted += 1

print(f"Converted {converted} evidence objects into fusion-ready format.")
print(f"Saved to: {output_folder}/")

# ------------------------------------------------------------
# 3. SHOW ONE EXAMPLE
# ------------------------------------------------------------
if evidence_files:
    with open(glob.glob(f"{output_folder}/*.json")[0]) as f:
        example = json.load(f)
    print("\nExample Fusion Evidence:")
    print(json.dumps(example, indent=2))