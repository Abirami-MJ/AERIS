import pandas as pd
import json
import numpy as np

# ==========================================
# Load Metadata
# ==========================================

metadata = pd.read_csv("NTSB_metadata.csv")

print("Metadata Loaded")
print(metadata.shape)

# ==========================================
# Load Similarity Scores (Optional)
# ==========================================

try:
    similarity = np.load("NTSB_similarity_scores.npy")

except:

    similarity = np.ones(len(metadata))

# ==========================================
# Create Evidence Objects
# ==========================================

evidence_objects = []

for i, row in metadata.iterrows():

    evidence = {

        "source": "NTSB",

        "event_id": row["event_id"],

        "aircraft_make": row["aircraft_make"],

        "aircraft_model": row["aircraft_model"],

        "aircraft_category": row["aircraft_category"],

        "engine_type": row["engine_type"],

        "flight_phase": row["flight_phase"],

        "weather": row["weather"],

        "damage": row["damage"],

        "purpose": row["purpose"],

        "fatal_injuries": int(row["fatal_injuries"]),

        "serious_injuries": int(row["serious_injuries"]),

        "minor_injuries": int(row["minor_injuries"]),

        "probable_cause": row["probable_cause"],

        "findings": row["findings"],

        "retrieval_confidence": float(similarity[i]),

        "report_text": row["report_text"]

    }

    evidence_objects.append(evidence)

# ==========================================
# Save JSON
# ==========================================

with open("NTSB_Evidence_Object.json", "w", encoding="utf-8") as f:

    json.dump(evidence_objects, f, indent=4)

print("\nEvidence Object Saved")

# ==========================================
# Save CSV
# ==========================================

pd.DataFrame(evidence_objects).to_csv(

    "NTSB_Evidence_Object.csv",

    index=False

)

print("CSV Saved")

# ==========================================
# Statistics
# ==========================================

print("\nSummary")
print("----------------------------")
print("Evidence Objects :", len(evidence_objects))

print("Fields per Object:", len(evidence_objects[0]))

print("\nFiles Generated")
print("----------------------------")
print("1. NTSB_Evidence_Object.json")
print("2. NTSB_Evidence_Object.csv")

print("\nEvidence Generation Completed Successfully!")