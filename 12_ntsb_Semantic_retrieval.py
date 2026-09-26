# ==========================================================
# AERIS
# NTSB Semantic Retrieval
# ==========================================================

import os
import json
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ==========================================================
# File Paths
# ==========================================================

CMAPSS_EVIDENCE = "evidence_objects/cmapss_evidence_97.json"

METADATA_FILE = "NTSB_metadata.csv"

FAISS_INDEX = "NTSB_FAISS.index"

OUTPUT_FOLDER = "Semantic_Retrieval"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "retrieved_ntsb_case.json"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Load CMAPSS Evidence
# ==========================================================

print("="*70)
print("Loading CMAPSS Evidence Object")
print("="*70)

with open(CMAPSS_EVIDENCE, "r", encoding="utf-8") as f:
    cmapss = json.load(f)

# Prefer FAISS query
query = cmapss.get("faiss_query")

if query is None:
    query = cmapss.get("semantic_query")

if query is None:
    raise ValueError("No semantic query found inside CMAPSS evidence object.")

print("\nSemantic Query")
print("----------------------------")
print(query)

# ==========================================================
# Load Metadata
# ==========================================================

metadata = pd.read_csv(METADATA_FILE)

print("\nMetadata Loaded")
print(metadata.shape)

# ==========================================================
# Load FAISS Index
# ==========================================================

index = faiss.read_index(FAISS_INDEX)

print("\nFAISS Loaded")
print("Vectors :", index.ntotal)

# ==========================================================
# Load Embedding Model
# ==========================================================

print("\nLoading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded")

# ==========================================================
# Encode Query
# ==========================================================

query_embedding = model.encode(
    query,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

query_embedding = np.expand_dims(query_embedding, axis=0)

# ==========================================================
# Semantic Search
# ==========================================================

TOP_K = 5

scores, indices = index.search(query_embedding, TOP_K)

print("\n")
print("="*70)
print("TOP RETRIEVED NTSB CASES")
print("="*70)

retrieved_results = []

for rank, idx in enumerate(indices[0]):

    row = metadata.iloc[idx]

    similarity = float(scores[0][rank])

    result = {

        "rank": rank + 1,

        "similarity": round(similarity,4),

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

        "report_text": row["report_text"]

    }

    retrieved_results.append(result)

    print("="*70)
    print(f"Rank        : {rank+1}")
    print(f"Similarity  : {similarity:.4f}")
    print(f"Event ID    : {row['event_id']}")
    print(f"Aircraft    : {row['aircraft_make']} {row['aircraft_model']}")
    print("\nProbable Cause:")
    print(row["probable_cause"])
    print("\nFindings:")
    print(row["findings"])

# ==========================================================
# Save Top-1 Retrieved Case
# ==========================================================

top_case = {

    "retrieval_query": query,

    "retrieval_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),

    "top_match": retrieved_results[0]

}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        top_case,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n")
print("="*70)
print("Top-1 Retrieved Case Saved")
print("="*70)

print(f"Location : {OUTPUT_FILE}")

print("\nPipeline Completed Successfully")