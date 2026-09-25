# ============================================================
# AERIS - NTSB DATA PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np
import re

print("="*60)
print("AERIS - NTSB PREPROCESSING")
print("="*60)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv(
    "final_reports_2016-23_cons_2024-12-24.csv",
    sep=';',
    encoding='utf-8',
    low_memory=False
)

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)

# ------------------------------------------------------------
# Drop Unnecessary Columns
# ------------------------------------------------------------

drop_cols = [
    "ReportNo",
    "EventID",
    "RepGenFlag",
    "Scheduled",
    "Operator",
    "HighestInjuryLevel",
    "Mode",
    "ReportStatus",
    "MostRecentReportType"
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

print("\nDropped unnecessary columns")

# ------------------------------------------------------------
# Convert Date
# ------------------------------------------------------------

df["EventDate"] = pd.to_datetime(
    df["EventDate"],
    errors="coerce"
)

# ------------------------------------------------------------
# Handle Missing Values
# ------------------------------------------------------------

text_fill_unknown = [
    "State",
    "AirportID",
    "AirportName",
    "PurposeOfFlight",
    "EngineType",
    "AirCraftDamage"
]

for col in text_fill_unknown:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Number of Engines

if "NumberOfEngines" in df.columns:
    mode = df["NumberOfEngines"].mode()[0]
    df["NumberOfEngines"] = df["NumberOfEngines"].fillna(mode)

# Injury Columns

injury_cols = [
    "FatalInjuryCount",
    "SeriousInjuryCount",
    "MinorInjuryCount",
    "OnboardInjuryCount",
    "OnGroundInjuryCount"
]

for col in injury_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print("\nMissing values handled")

# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

print("\nCreating New Features...")

df["EventYear"] = df["EventDate"].dt.year

df["EventMonth"] = df["EventDate"].dt.month

df["Weekday"] = df["EventDate"].dt.day_name()

df["TotalInjuries"] = (
      df["FatalInjuryCount"]
    + df["SeriousInjuryCount"]
    + df["MinorInjuryCount"]
)

df["TotalAffectedPeople"] = (
      df["OnboardInjuryCount"]
    + df["OnGroundInjuryCount"]
)

# Report Length

df["ReportLength"] = (
    df["rep_text"]
    .fillna("")
    .astype(str)
    .str.len()
)

# Probable Cause Length

df["CauseLength"] = (
    df["ProbableCause"]
    .fillna("")
    .astype(str)
    .str.len()
)

# Findings Length

df["FindingsLength"] = (
    df["Findings"]
    .fillna("")
    .astype(str)
    .str.len()
)

print("Feature engineering completed")

# ------------------------------------------------------------
# Text Cleaning
# ------------------------------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # remove html
    text = re.sub("<.*?>", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # remove strange characters
    text = re.sub(r"[^\w\s.,;:!?()/\-]", " ", text)

    return text.strip()

text_cols = [
    "ProbableCause",
    "Findings",
    "rep_text"
]

for col in text_cols:
    df[col] = df[col].apply(clean_text)

print("Text cleaned")

# ------------------------------------------------------------
# Duplicate Check
# ------------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()

# ------------------------------------------------------------
# Missing Value Check
# ------------------------------------------------------------

missing = df.isnull().sum()

missing = missing[missing > 0]

print("\nRemaining Missing Values")

if len(missing)==0:
    print("No Missing Values")

else:
    print(missing)

# ------------------------------------------------------------
# Save Dataset
# ------------------------------------------------------------

df.to_csv(
    "NTSB_cleaned.csv",
    index=False
)

print("\nCleaned Dataset Saved")

print("Final Shape :", df.shape)

print("="*60)
print("PREPROCESSING COMPLETED")
print("="*60)