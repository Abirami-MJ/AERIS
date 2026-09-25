"""
AERIS - Post-Cleaning Quick Check (C-MAPSS FD001)
================================================
Run this AFTER cleaning_cmapss.py has been run with DATASET_TYPE = "cmapss"
Just validates the cleaning worked - NOT a full EDA re-run.
"""

import pandas as pd

# Path to your cleaned C-MAPSS file
CLEANED_FILE = "cleaned_outputs/C-MAPSS_FD001/C-MAPSS_FD001_cleaned.csv"

df = pd.read_csv(CLEANED_FILE)

print("="*60)
print("POST-CLEANING CHECK: C-MAPSS FD001")
print("="*60)

# 1. Shape check
print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("(Original raw FD001 train had 20,631 rows x 26 columns -"
      " columns may be fewer now if constant sensors were dropped)")

# 2. Duplicates - should be 0
print(f"\nDuplicate rows: {df.duplicated().sum()} (should be 0)")

# 3. Missing values - check what's left
print(f"\nColumns with remaining missing values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if not missing.empty else "None - all handled")

# 4. Confirm critical columns have no nulls
critical_cols = ["unit_number", "time_cycle"]
for col in critical_cols:
    if col in df.columns:
        n_null = df[col].isnull().sum()
        print(f"  '{col}': {n_null} nulls (should be 0)")

# 5. Confirm constant/zero-variance sensors were actually dropped
possible_constant_sensors = ["epr", "farB", "Nf_dmd", "PCNfR_dmd"]
still_present = [c for c in possible_constant_sensors if c in df.columns]
dropped = [c for c in possible_constant_sensors if c not in df.columns]
print(f"\nPossible constant sensors still present: {still_present}")
print(f"Possible constant sensors dropped: {dropped}")

# 6. Data types check
print(f"\nData types check (should all be numeric):")
print(df.dtypes)

# 7. Engine count check
if "unit_number" in df.columns:
    n_engines = df["unit_number"].nunique()
    print(f"\nUnique engines: {n_engines} (FD001 train should have 100)")

# 8. Cycle range check
if "time_cycle" in df.columns:
    print(f"\nCycle range: {df['time_cycle'].min()} to {df['time_cycle'].max()}")

# 9. Outlier flag columns - quick summary
outlier_cols = [c for c in df.columns if c.endswith("_is_outlier")]
if outlier_cols:
    print(f"\nOutlier flag columns created: {len(outlier_cols)}")
    for col in outlier_cols:
        n_flagged = df[col].sum()
        if n_flagged > 0:
            print(f"  {col}: {n_flagged} flagged ({n_flagged/len(df)*100:.2f}%)")

# 10. Quick describe of sensor ranges (sanity check for scale/units)
numeric_cols = df.select_dtypes(include='number').columns
print(f"\nQuick numeric summary:")
print(df[numeric_cols].describe().T[['min', 'max', 'mean']])

print("\n" + "="*60)
print("CHECK COMPLETE")
print("="*60)