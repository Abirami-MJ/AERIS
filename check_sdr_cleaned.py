"""
AERIS - Post-Cleaning Quick Check (SDR)
================================================
Run this AFTER cleaning_cmapss.py has been run with DATASET_TYPE = "sdr"
Just validates the cleaning worked - NOT a full EDA re-run.
"""

import pandas as pd

# Path to your cleaned SDR file
CLEANED_FILE = "cleaned_outputs/SDR/SDR_cleaned.csv"

df = pd.read_csv(CLEANED_FILE)

print("="*60)
print("POST-CLEANING CHECK: SDR")
print("="*60)

# 1. Shape check
print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# 2. Duplicates - should be 0
print(f"\nDuplicate rows: {df.duplicated().sum()} (should be 0)")

# 3. Missing values - check what's left
print(f"\nColumns with remaining missing values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if not missing.empty else "None - all handled")

# 4. Confirm low-priority structural columns were dropped
dropped_check = ["FuselageStationFrom", "WingStationFrom", "ButtLineFrom", "WaterLineFrom"]
still_present = [c for c in dropped_check if c in df.columns]
print(f"\nLow-priority columns still present (should be empty list): {still_present}")

# 5. Confirm critical columns have no nulls
critical_cols = ["RegistryNNumber", "DifficultyDate", "NatureOfConditionA", "Discrepancy"]
for col in critical_cols:
    if col in df.columns:
        n_null = df[col].isnull().sum()
        print(f"  '{col}': {n_null} nulls (should be 0)")

# 6. Data types - confirm dates and numerics converted correctly
print(f"\nData types check:")
print(df[["DifficultyDate"]].dtypes if "DifficultyDate" in df.columns else "DifficultyDate not found")

# 7. Outlier flag columns - quick summary
outlier_cols = [c for c in df.columns if c.endswith("_is_outlier")]
if outlier_cols:
    print(f"\nOutlier flag columns created: {len(outlier_cols)}")
    for col in outlier_cols:
        n_flagged = df[col].sum()
        if n_flagged > 0:
            print(f"  {col}: {n_flagged} flagged")

# 8. Text standardization check - sample values
if "NatureOfConditionA" in df.columns:
    print(f"\nSample standardized values (NatureOfConditionA):")
    print(df["NatureOfConditionA"].dropna().unique()[:10])

print("\n" + "="*60)
print("CHECK COMPLETE")
print("="*60)