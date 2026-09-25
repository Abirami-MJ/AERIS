# ==========================================================
# AERIS - NTSB Embedding Generation
# ==========================================================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================================================
# Load Metadata
# ==========================================================

print("=" * 60)
print("Loading NTSB Metadata...")
print("=" * 60)

metadata = pd.read_csv("NTSB_metadata.csv")

print(f"Metadata Shape : {metadata.shape}")

# ==========================================================
# Check Required Columns
# ==========================================================

required_columns = ["event_id", "document"]

for col in required_columns:
    if col not in metadata.columns:
        raise ValueError(f"Missing Required Column: {col}")

print("Required columns found.")

# ==========================================================
# Handle Missing Documents
# ==========================================================

metadata["document"] = metadata["document"].fillna("").astype(str)

# ==========================================================
# Load Sentence Transformer
# ==========================================================

print("\nLoading Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully")

# ==========================================================
# Generate Embeddings
# ==========================================================

print("\nGenerating Embeddings...")

embeddings = []

for text in tqdm(metadata["document"], total=len(metadata)):

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings.append(embedding)

embeddings = np.array(embeddings)

print("\nEmbedding Generation Completed")
print(f"Embedding Shape : {embeddings.shape}")

# Expected:
# (7462, 384)

# ==========================================================
# Save Embeddings
# ==========================================================

np.save("NTSB_embeddings.npy", embeddings)

print("Embeddings Saved Successfully")

# ==========================================================
# Save Event IDs
# ==========================================================

metadata[["event_id"]].to_csv(
    "NTSB_embedding_ids.csv",
    index=False
)

print("Embedding IDs Saved Successfully")

# ==========================================================
# Verification
# ==========================================================

assert len(metadata) == len(embeddings)

print("\nVerification Passed")
print(f"Total Documents  : {len(metadata)}")
print(f"Total Embeddings : {len(embeddings)}")

print("\nFiles Generated")
print("----------------------------")
print("1. NTSB_embeddings.npy")
print("2. NTSB_embedding_ids.csv")

print("\nEmbedding Phase Completed Successfully!")