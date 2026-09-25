# ==========================================================
# AERIS - NTSB Semantic Retrieval
# Uses CMAPSS Evidence Object
# ==========================================================

import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================================================
# Load Files
# ==========================================================

print("=" * 60)
print("Loading Files...")
print("=" * 60)

metadata = pd.read_csv("NTSB_metadata.csv")

index = faiss.read_index("NTSB_FAISS.index")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Metadata Loaded :", len(metadata))
print("FAISS Index Loaded")
print("Sentence Transformer Loaded")

# ==========================================================
# Load CMAPSS Evidence Object
# ==========================================================

print("\nLoading CMAPSS Evidence Object...")

with open(
    "evidence_objects/cmapss_evidence_97.json",
    "r",
    encoding="utf-8"
) as f:
    cmapss = json.load(f)

# ==========================================================
# Build Semantic Query
# ==========================================================

failure = cmapss["known_failure_mechanism"]

features = ", ".join(cmapss["shap_top_features"])

query = (
    f"{failure} degradation "
    f"related to {features} "
    f"failure"
)

print("\nGenerated Query")
print(query)

# ==========================================================
# Encode Query
# ==========================================================

query_embedding = model.encode(
    query,
    convert_to_numpy=True,
    normalize_embeddings=True
)

query_embedding = np.expand_dims(query_embedding, axis=0)

# ==========================================================
# Semantic Search
# ==========================================================

TOP_K = 5

scores, indices = index.search(query_embedding, TOP_K)

print("\nTop Retrieved Reports")

retrieved_reports = []

for rank, idx in enumerate(indices[0]):

    similarity = float(scores[0][rank])

    row = metadata.iloc[idx]

    report = {

        "rank": rank + 1,

        "similarity_score": round(similarity,4),

        "event_id": row["event_id"],

        "aircraft_make": row["aircraft_make"],

        "aircraft_model": row["aircraft_model"],

        "probable_cause": row["probable_cause"],

        "findings": row["findings"],

        "report_text": row["report_text"]

    }

    retrieved_reports.append(report)

    print("=" * 70)

    print("Rank :", rank + 1)

    print("Similarity :", round(similarity,4))

    print("\nEvent ID")
    print(row["event_id"])

    print("\nAircraft")
    print(row["aircraft_make"], row["aircraft_model"])

    print("\nProbable Cause")
    print(row["probable_cause"])

    print("\nFindings")
    print(row["findings"])

# ==========================================================
# Save Retrieval Output
# ==========================================================

output = {

    "query": query,

    "retrieval_model": "all-MiniLM-L6-v2 + FAISS",

    "top_k": TOP_K,

    "results": retrieved_reports

}

with open(
    "retrieved_ntsb_reports.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(output, f, indent=4)

print("\nRetrieved Reports Saved Successfully")

print("File : retrieved_ntsb_reports.json")

print("\nSemantic Retrieval Completed!")