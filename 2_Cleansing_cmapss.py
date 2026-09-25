"""
AERIS - Data Cleaning Script (C-MAPSS FD001)
================================================
Run this AFTER EDA has been reviewed for this dataset.

Cleaning steps performed:
  1. Duplicate removal
  2. Constant/zero-variance sensor detection (common in C-MAPSS - flat sensors
     add no signal and are dropped)
  3. Data type validation (numeric columns)
  4. Missing value handling
  5. Outlier flagging (NOT removed - flagged, since late-cycle "outliers"
     are often real degradation signal, not noise)
  6. Cleaned dataset saved as CSV automatically
  7. Full cleaning log saved as a text file

Supports re-use for other datasets by changing DATASET_TYPE below
(configs for 'sdr', 'ngafid', 'ntsb' also included).
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION - CHANGE THIS FOR EACH DATASET
# =========================================================
"""
DATASET_TYPE = "cmapss"
FILE_PATH = "train_FD001_labeled.csv"
DATASET_NAME = "C-MAPSS_FD001" 
"""

DATASET_TYPE = "sdr"
FILE_PATH = "Aircraft_failure_SDR-2024-FAA.csv"
DATASET_NAME = "SDR"

# =========================================================
# CLEANING CONFIG
# =========================================================
CLEANING_CONFIG = {
    "cmapss": {
        "id_col": "unit_number",
        "time_col": "time_cycle",
        "date_cols": [],
        "numeric_cols": ["time_cycle", "op_setting_1", "op_setting_2", "op_setting_3",
                          "T2", "T24", "T30", "T50", "P2", "P15", "P30",
                          "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc",
                          "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"],
        "text_cols_to_standardize": [],
        "narrative_col": None,
        "critical_cols": ["unit_number", "time_cycle"],
        "low_priority_drop_cols": [],
        "auto_drop_constant_cols": True,   # drops zero-variance sensor columns automatically
    },
    "sdr": {
        "id_col": "RegistryNNumber",
        "date_cols": ["DifficultyDate", "SubmissionDate"],
        "numeric_cols": ["AircraftTotalTime", "AircraftTotalCycles",
                          "EngineTotalTime", "EngineTotalCycles",
                          "PartTotalTime", "PartTotalCycles",
                          "ComponentTotalTime", "ComponentTotalCycles",
                          "CrackLength", "NumberOfCracks"],
        "text_cols_to_standardize": ["AircraftMake", "AircraftModel",
                                      "EngineMake", "EngineModel",
                                      "PartMake", "PartName", "PartCondition",
                                      "ComponentMake", "ComponentName",
                                      "NatureOfConditionA", "NatureOfConditionB",
                                      "NatureOfConditionC"],
        "narrative_col": "Discrepancy",
        "critical_cols": ["RegistryNNumber", "DifficultyDate",
                           "NatureOfConditionA", "Discrepancy"],
        "low_priority_drop_cols": ["FuselageStationFrom", "FuselageStationTo",
                                    "StringerFrom", "StringerFromSide",
                                    "StringerTo", "StringerToSide",
                                    "WingStationFrom", "WingStationFromSide",
                                    "WingStationTo", "WingStationToSide",
                                    "ButtLineFrom", "ButtLineFromSide",
                                    "ButtlineTo", "ButtlineToSide",
                                    "WaterLineFrom", "WaterLineTo"],
        "auto_drop_constant_cols": False,
    },
    "ngafid": {
        "id_col": "aircraft_id",
        "date_cols": ["event_date"],
        "numeric_cols": [],
        "text_cols_to_standardize": [],
        "narrative_col": "discrepancy_description",
        "critical_cols": ["aircraft_id", "event_date"],
        "low_priority_drop_cols": [],
        "auto_drop_constant_cols": False,
    },
    "ntsb": {
        "id_col": "event_id",
        "date_cols": ["event_date"],
        "numeric_cols": [],
        "text_cols_to_standardize": ["aircraft_make_model", "component/system", "event_type"],
        "narrative_col": "narrative",
        "critical_cols": ["event_id", "event_date", "probable_cause"],
        "low_priority_drop_cols": [],
        "auto_drop_constant_cols": False,
    },
}

# =========================================================
# SETUP OUTPUT FOLDER
# =========================================================
OUTPUT_DIR = os.path.join("cleaned_outputs", DATASET_NAME.replace(" ", "_"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG = []


def log(msg):
    print(msg)
    LOG.append(str(msg))


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(filepath):
    log(f"\n{'='*70}\nLOADING DATASET: {DATASET_NAME}\n{'='*70}")
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        log("UTF-8 failed, retrying with latin1 encoding...")
        df = pd.read_csv(filepath, encoding='latin1')
    log(f"Initial shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# =========================================================
# 2. REMOVE DUPLICATES
# =========================================================
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    log(f"\n[Duplicates] Removed {removed} duplicate rows "
        f"({removed/before*100:.2f}% of {before})")
    return df


# =========================================================
# 3. DROP LOW-PRIORITY / IRRELEVANT COLUMNS
# =========================================================
def drop_low_priority_columns(df, config):
    cols_to_drop = [c for c in config.get("low_priority_drop_cols", []) if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log(f"\n[Column Drop] Removed {len(cols_to_drop)} low-priority columns: {cols_to_drop}")
    return df


# =========================================================
# 4. DROP CONSTANT / ZERO-VARIANCE COLUMNS (key for C-MAPSS)
# =========================================================
def drop_constant_columns(df, config):
    if not config.get("auto_drop_constant_cols", False):
        return df

    numeric_cols = [c for c in config.get("numeric_cols", []) if c in df.columns]
    constant_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]

    if constant_cols:
        df = df.drop(columns=constant_cols)
        log(f"\n[Constant Columns] Dropped {len(constant_cols)} zero-variance "
            f"columns (no signal): {constant_cols}")
    else:
        log(f"\n[Constant Columns] None found - all sensors show variation.")
    return df


# =========================================================
# 5. FIX DATA TYPES (dates, numerics)
# =========================================================
def fix_data_types(df, config):
    log(f"\n[Data Types] Validating date and numeric columns...")

    for col in config.get("date_cols", []):
        if col in df.columns:
            before_na = df[col].isnull().sum()
            df[col] = pd.to_datetime(df[col], errors='coerce')
            after_na = df[col].isnull().sum()
            log(f"  '{col}' -> datetime | newly unparseable: {after_na - before_na}")

    for col in config.get("numeric_cols", []):
        if col in df.columns:
            before_na = df[col].isnull().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_na = df[col].isnull().sum()
            if after_na > before_na:
                log(f"  '{col}' -> numeric | newly unparseable: {after_na - before_na}")

    return df


# =========================================================
# 6. STANDARDIZE TEXT / CATEGORICAL COLUMNS
# =========================================================
def standardize_text_columns(df, config):
    cols = config.get("text_cols_to_standardize", [])
    if not cols:
        return df
    log(f"\n[Text Standardization] Cleaning categorical columns...")
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({'NAN': np.nan, 'NONE': np.nan, '': np.nan})
            )
            log(f"  '{col}' -> trimmed, uppercased, blanks normalized to NaN")
    return df


# =========================================================
# 7. CLEAN NARRATIVE / TEXT FIELD
# =========================================================
def clean_narrative_column(df, config):
    col = config.get("narrative_col")
    if col and col in df.columns:
        log(f"\n[Narrative Cleaning] Processing '{col}'...")
        before_empty = (df[col].astype(str).str.strip() == '').sum()
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, '': np.nan})
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
        after_null = df[col].isnull().sum()
        log(f"  Empty/whitespace-only narratives found: {before_empty}, now null: {after_null}")
    return df


# =========================================================
# 8. HANDLE MISSING VALUES
# =========================================================
def handle_missing_values(df, config):
    log(f"\n[Missing Values] Handling nulls...")

    critical_cols = [c for c in config.get("critical_cols", []) if c in df.columns]
    if critical_cols:
        before = len(df)
        df = df.dropna(subset=critical_cols)
        removed = before - len(df)
        log(f"  Dropped {removed} rows missing critical fields {critical_cols}")

    numeric_cols = [c for c in config.get("numeric_cols", []) if c in df.columns]
    any_missing = False
    for col in numeric_cols:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 0:
            any_missing = True
            log(f"  '{col}': {missing_pct:.2f}% missing "
                f"(left as NaN - impute during feature engineering, not here)")
    if not any_missing:
        log(f"  No missing values found in numeric columns.")

    text_cols = config.get("text_cols_to_standardize", [])
    for col in text_cols:
        if col in df.columns and col not in critical_cols:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                df[col] = df[col].fillna("UNKNOWN")
                log(f"  '{col}': filled {n_missing} missing values with 'UNKNOWN'")

    return df


# =========================================================
# 9. FLAG OUTLIERS (does NOT remove - just flags for review)
# =========================================================
def flag_outliers(df, config):
    numeric_cols = [c for c in config.get("numeric_cols", []) if c in df.columns]
    if not numeric_cols:
        return df

    log(f"\n[Outlier Flagging] Checking numeric columns (IQR method)...")
    outlier_flags = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue  # skip constant-like columns
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flag_col = f"{col}_is_outlier"
            outlier_flags[flag_col] = (df[col] < lower) | (df[col] > upper)
            n_out = outlier_flags[flag_col].sum()
            if n_out > 0:
                log(f"  '{col}': {n_out} outliers flagged "
                    f"({n_out/len(df)*100:.2f}%) - NOT removed")

    df = pd.concat([df, outlier_flags], axis=1)
    return df


# =========================================================
# 10. FINAL VALIDATION
# =========================================================
def final_validation(df, original_shape):
    log(f"\n{'='*70}\nFINAL VALIDATION - {DATASET_NAME}\n{'='*70}")
    log(f"Original shape: {original_shape[0]:,} rows x {original_shape[1]} columns")
    log(f"Cleaned shape:  {df.shape[0]:,} rows x {df.shape[1]} columns")
    log(f"Rows removed: {original_shape[0] - df.shape[0]:,} "
        f"({(original_shape[0] - df.shape[0])/original_shape[0]*100:.2f}%)")
    log(f"Columns removed: {original_shape[1] - (df.shape[1] - len([c for c in df.columns if c.endswith('_is_outlier')]))}")
    log(f"Remaining duplicate rows: {df.duplicated().sum()}")
    log(f"Columns with remaining nulls: {df.isnull().any().sum()}")


# =========================================================
# 11. SAVE CLEANED DATA + LOG
# =========================================================
def save_outputs(df):
    csv_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_cleaned.csv")
    df.to_csv(csv_path, index=False)
    log(f"\n✓ Cleaned dataset saved to: {os.path.abspath(csv_path)}")

    log_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_cleaning_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print(f"✓ Cleaning log saved to: {os.path.abspath(log_path)}")


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    config = CLEANING_CONFIG[DATASET_TYPE]

    df = load_data(FILE_PATH)
    original_shape = df.shape

    df = remove_duplicates(df)
    df = drop_low_priority_columns(df, config)
    df = drop_constant_columns(df, config)
    df = fix_data_types(df, config)
    df = standardize_text_columns(df, config)
    df = clean_narrative_column(df, config)
    df = handle_missing_values(df, config)
    df = flag_outliers(df, config)

    final_validation(df, original_shape)
    save_outputs(df)
