import pandas as pd
import json

# ==========================================================
# Load Cleaned Dataset
# ==========================================================

df = pd.read_csv("NTSB_cleaned.csv")

print("Dataset Loaded")
print(df.shape)

# ==========================================================
# Select Metadata Columns
# ==========================================================

metadata = df[[
    "NtsbNo",
    "EventDate",
    "Make",
    "Model",
    "AirCraftCategory",
    "EngineType",
    "NumberOfEngines",
    "BroadPhaseofFlight",
    "WeatherCondition",
    "AirCraftDamage",
    "PurposeOfFlight",
    "FatalInjuryCount",
    "SeriousInjuryCount",
    "MinorInjuryCount",
    "ProbableCause",
    "Findings",
    "rep_text"
]].copy()

# ==========================================================
# Rename Columns
# ==========================================================

metadata.rename(columns={
    "NtsbNo":"event_id",
    "EventDate":"event_date",
    "Make":"aircraft_make",
    "Model":"aircraft_model",
    "AirCraftCategory":"aircraft_category",
    "EngineType":"engine_type",
    "NumberOfEngines":"num_engines",
    "BroadPhaseofFlight":"flight_phase",
    "WeatherCondition":"weather",
    "AirCraftDamage":"damage",
    "PurposeOfFlight":"purpose",
    "FatalInjuryCount":"fatal_injuries",
    "SeriousInjuryCount":"serious_injuries",
    "MinorInjuryCount":"minor_injuries",
    "ProbableCause":"probable_cause",
    "Findings":"findings",
    "rep_text":"report_text"
}, inplace=True)

# ==========================================================
# Fill Remaining Missing Values
# ==========================================================

text_columns = [
    "engine_type",
    "flight_phase",
    "purpose",
    "damage",
    "probable_cause",
    "findings",
    "report_text"
]

for col in text_columns:
    metadata[col] = metadata[col].fillna("Unknown")

num_columns = [
    "fatal_injuries",
    "serious_injuries",
    "minor_injuries"
]

for col in num_columns:
    metadata[col] = metadata[col].fillna(0)

# ==========================================================
# Create Search Document
# (Used Later for Embeddings & RAG)
# ==========================================================

metadata["document"] = (
    "Aircraft Make: " + metadata["aircraft_make"].astype(str) +
    ". Model: " + metadata["aircraft_model"].astype(str) +
    ". Engine: " + metadata["engine_type"].astype(str) +
    ". Flight Phase: " + metadata["flight_phase"].astype(str) +
    ". Weather: " + metadata["weather"].astype(str) +
    ". Damage: " + metadata["damage"].astype(str) +
    ". Purpose: " + metadata["purpose"].astype(str) +
    ". Probable Cause: " + metadata["probable_cause"].astype(str) +
    ". Findings: " + metadata["findings"].astype(str) +
    ". Investigation Report: " + metadata["report_text"].astype(str)
)

# ==========================================================
# Save Metadata CSV
# ==========================================================

metadata.to_csv("ntsb_metadata.csv", index=False)

print("Metadata CSV Saved")

# ==========================================================
# Save JSON
# ==========================================================

metadata.to_json(
    "ntsb_metadata.json",
    orient="records",
    indent=4
)

print("Metadata JSON Saved")

# ==========================================================
# Display Sample
# ==========================================================

print("\nMetadata Shape :", metadata.shape)

print("\nSample Metadata")
print(metadata.head())