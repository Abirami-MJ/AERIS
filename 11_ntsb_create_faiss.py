# ==========================================================
# AERIS - NTSB FAISS Index Creation
# ==========================================================

import numpy as np
import pandas as pd
import faiss
import os

# ==========================================================
# Load Embeddings
# ==========================================================

print("=" * 60)
print("Loading Embeddings...")
print("=" * 60)

embeddings = np.load("NTSB_embeddings.npy")

print(f"Embedding Shape : {embeddings.shape}")

# ==========================================================
# Load Metadata IDs
# ==========================================================

metadata = pd.read_csv("NTSB_embedding_ids.csv")

print(f"Metadata Records : {len(metadata)}")

# ==========================================================
# Verification
# ==========================================================

assert len(metadata) == embeddings.shape[0]

print("Verification Passed")

# ==========================================================
# Create FAISS Index
# ==========================================================

dimension = embeddings.shape[1]

print(f"\nEmbedding Dimension : {dimension}")

# Cosine similarity
index = faiss.IndexFlatIP(dimension)

# Embeddings are already normalized
index.add(embeddings.astype("float32"))

print("\nFAISS Index Created")

print(f"Total Stored Vectors : {index.ntotal}")

# ==========================================================
# Save Index
# ==========================================================

faiss.write_index(index, "NTSB_FAISS.index")

print("\nFAISS Index Saved Successfully")

# ==========================================================
# Verification
# ==========================================================

loaded_index = faiss.read_index("NTSB_FAISS.index")

print("\nVerification Passed")

print(f"Stored Vectors : {loaded_index.ntotal}")

print("\nGenerated File")
print("----------------------------")
print("NTSB_FAISS.index")

print("\nFAISS Phase Completed Successfully!")