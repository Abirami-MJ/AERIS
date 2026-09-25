import json
import os

# ==========================================================
# CHANGE THIS PATH TO YOUR ACTUAL NTSB EVIDENCE FILE
# ==========================================================

INPUT_FILE = "NTSB_Evidence_Object.json"

# Example:
# INPUT_FILE = "NTSB/ntsb_evidence_objects.json"
# INPUT_FILE = "evidence_objects/ntsb_evidence.json"

# ==========================================================
# Load Evidence
# ==========================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    evidence = json.load(f)

# If only one object, convert to list
if isinstance(evidence, dict):
    evidence = [evidence]

print(f"Loaded {len(evidence)} evidence objects")

# ==========================================================
# Convert to Fusion Evidence
# ==========================================================

fusion_list = []

for item in evidence:

    damage = str(item.get("damage", "")).lower()

    fatal = int(item.get("fatal_injuries", 0))
    serious = int(item.get("serious_injuries", 0))

    confidence = float(item.get("retrieval_confidence", 1.0))

    # ----------------------------
    # Rule-Based Hypothesis
    # ----------------------------

    if fatal > 0:
        hypothesis = "High Risk"

    elif serious > 0:
        hypothesis = "High Risk"

    elif "substantial" in damage:
        hypothesis = "Medium Risk"

    elif "minor" in damage:
        hypothesis = "Minor Risk"

    else:
        hypothesis = "Healthy"

    fusion = {

        "source": "NTSB",

        "event_id": item.get("event_id"),

        "hypothesis": hypothesis,

        "belief": round(confidence, 3),

        "uncertainty": round(1 - confidence, 3)

    }

    fusion_list.append(fusion)

# ==========================================================
# Save Fusion Evidence
# ==========================================================

OUTPUT_FOLDER = "Fusion_Evidence/NTSB"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "ntsb_fusion_evidence.json"
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(fusion_list, f, indent=4)

print("\nFusion Evidence Saved Successfully!")

print(f"Saved at : {OUTPUT_FILE}")

print(f"Total Fusion Objects : {len(fusion_list)}")

print("\nExample:\n")

print(json.dumps(fusion_list[0], indent=4))