"""
AERIS - Professional EDA Script (Per-Dataset)
================================================
Run this SEPARATELY for each dataset by changing DATASET_TYPE and FILE_PATH below.
Supports: 'cmapss', 'ngafid', 'sdr', 'ntsb'

Outputs:
  - Console insights report
  - All graphs saved to: eda_outputs/<dataset_name>/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION - CHANGE THIS FOR EACH DATASET
# =========================================================
"""
DATASET_TYPE = "sdr"          # 'cmapss' | 'ngafid' | 'sdr' | 'ntsb'
FILE_PATH = "Aircraft_failure_SDR-2024-FAA.csv"   # path to your CSV file
DATASET_NAME = "SDR"  """       # used for folder/report naming


DATASET_TYPE = "cmapss"
FILE_PATH = "train_FD001_labeled.csv"
DATASET_NAME = "C-MAPSS_FD001"   

# Column name hints - EDIT these to match your actual column names
"""
COLUMN_CONFIG = {
        "sdr": {
        "id_col": "RegistryNNumber",
        "time_col": "DifficultyDate",
        "target_col": "NatureOfConditionA",
        "aircraft_cols": ["AircraftMake", "AircraftModel", "AircraftSerialNumber",
                          "AircraftTotalTime", "AircraftTotalCycles"],
        "engine_cols": ["EngineMake", "EngineModel", "EngineSerialNumber",
                        "EngineTotalTime", "EngineTotalCycles"],
        "part_cols": ["PartMake", "PartName", "PartNumber", "PartCondition",
                      "PartLocation", "PartTotalTime", "PartTotalCycles"],
        "component_cols": ["ComponentMake", "ComponentModel", "ComponentName",
                           "ComponentPartNumber", "ComponentLocation",
                           "ComponentTotalTime", "ComponentTotalCycles"],
        "condition_cols": ["NatureOfConditionA", "NatureOfConditionB", "NatureOfConditionC"],
        "narrative_col": "Discrepancy",
        "classification_col": "JASCCode",
        "structural_cols": ["FuselageStationFrom", "FuselageStationTo",
                            "StringerFrom", "StringerTo",
                            "WingStationFrom", "WingStationTo",
                            "ButtLineFrom", "ButtlineTo",
                            "WaterLineFrom", "WaterLineTo",
                            "CrackLength", "NumberOfCracks", "CorrosionLevel"],
        "drop_cols": ["OperatorControlNumber", "SubmissionDate", "OperatorDesignator",
                     "SubmitterDesignator", "SubmitterTypeCode", "ReceivingRegionCode",
                     "ReceivingDistrictOffice"],
    },
}"""


COLUMN_CONFIG={
        "cmapss": {
        "id_col": "unit_number",
        "time_col": "time_cycle",
        "target_col": None,   # RUL comes from a separate file (RUL_FD001.txt)
    },
}


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# =========================================================
# SETUP OUTPUT FOLDER
# =========================================================
OUTPUT_DIR = os.path.join("eda_outputs", DATASET_NAME.replace(" ", "_"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(filepath):
    print(f"\n{'='*70}\nLOADING DATASET: {DATASET_NAME}\n{'='*70}")
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumns:\n{list(df.columns)}")
    return df


# =========================================================
# 2. STRUCTURAL OVERVIEW
# =========================================================
def structural_overview(df):
    print(f"\n{'='*70}\nSTRUCTURAL OVERVIEW\n{'='*70}")
    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMemory usage:")
    print(f"  {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes} ({dupes/len(df)*100:.2f}%)")


# =========================================================
# 3. MISSING VALUES ANALYSIS + VISUALIZATION
# =========================================================
def missing_value_analysis(df):
    print(f"\n{'='*70}\nMISSING VALUES ANALYSIS\n{'='*70}")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
    report = report[report['missing_count'] > 0].sort_values('missing_pct', ascending=False)

    if report.empty:
        print("No missing values found.")
        return

    print(report)

    # Visualization 1: Missing value bar chart
    fig, ax = plt.subplots(figsize=(10, max(4, len(report) * 0.4)))
    ax.barh(report.index, report['missing_pct'], color='indianred')
    ax.set_xlabel("Missing %")
    ax.set_title(f"{DATASET_NAME}: Missing Values by Column")
    ax.invert_yaxis()
    save_plot(fig, "01_missing_values.png")

    # Visualization 2: Missing value heatmap (pattern check)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='Reds', ax=ax, yticklabels=False)
    ax.set_title(f"{DATASET_NAME}: Missing Value Pattern")
    save_plot(fig, "02_missing_value_heatmap.png")

    # INSIGHT
    high_missing = report[report['missing_pct'] > 30]
    if not high_missing.empty:
        print(f"\n⚠ INSIGHT: {len(high_missing)} column(s) have >30% missing data — "
              f"consider dropping or special imputation: {list(high_missing.index)}")


# =========================================================
# 4. NUMERIC FEATURE ANALYSIS
# =========================================================
def numeric_analysis(df):
    print(f"\n{'='*70}\nNUMERIC FEATURE ANALYSIS\n{'='*70}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("No numeric columns found.")
        return numeric_cols

    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print("\nSummary statistics:")
    print(df[numeric_cols].describe().T)

    # Visualization 1: Distribution grid (histograms)
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        df[col].hist(bins=40, ax=axes[i], color='steelblue', edgecolor='black')
        axes[i].set_title(col, fontsize=9)
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    save_plot(fig, "03_numeric_distributions.png")

    # Visualization 2: Boxplots (outlier detection)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
        axes[i].set_title(col, fontsize=9)
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    save_plot(fig, "04_boxplots_outliers.png")

    # Visualization 3: Correlation heatmap
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 0.6),
                                          max(6, len(numeric_cols) * 0.5)))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=len(numeric_cols) <= 15, fmt=".2f", cmap='coolwarm',
                    center=0, ax=ax, square=True)
        ax.set_title(f"{DATASET_NAME}: Correlation Matrix")
        save_plot(fig, "05_correlation_heatmap.png")

        # INSIGHT: highly correlated pairs
        corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        strong_corr = corr_pairs[abs(corr_pairs) > 0.8].sort_values(ascending=False)
        if not strong_corr.empty:
            print(f"\n⚠ INSIGHT: Strongly correlated feature pairs (|r| > 0.8) - "
                  f"candidates for redundancy removal:")
            print(strong_corr)

    # Outlier summary via IQR
    print("\nOutlier summary (IQR method):")
    outlier_summary = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            outlier_summary[col] = round(n_outliers / len(df) * 100, 2)
    for col, pct in sorted(outlier_summary.items(), key=lambda x: -x[1]):
        print(f"  {col}: {pct}% outliers")

    return numeric_cols


# =========================================================
# 5. CATEGORICAL / TEXT FEATURE ANALYSIS
# =========================================================
def categorical_analysis(df):
    print(f"\n{'='*70}\nCATEGORICAL / TEXT FEATURE ANALYSIS\n{'='*70}")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not cat_cols:
        print("No categorical columns found.")
        return cat_cols

    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"\n'{col}': {n_unique} unique values")
        if n_unique <= 20:
            print(df[col].value_counts())

    # Visualize top categories for low-cardinality columns (likely target/failure type cols)
    low_card_cols = [c for c in cat_cols if 2 <= df[c].nunique() <= 20]
    for col in low_card_cols[:6]:  # limit to 6 plots to avoid clutter
        fig, ax = plt.subplots(figsize=(10, 5))
        top_vals = df[col].value_counts().head(15)
        sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax, palette='viridis')
        ax.set_title(f"{DATASET_NAME}: Top Categories - {col}")
        ax.set_xlabel("Count")
        save_plot(fig, f"06_category_{col.replace('/', '_')}.png")

    # Text length analysis for narrative-style columns (SDR, NTSB)
    text_like_cols = [c for c in cat_cols if 'narrative' in c.lower()
                       or 'description' in c.lower() or 'cause' in c.lower()
                       or 'discrepancy' in c.lower()]

    for col in text_like_cols:
        lengths = df[col].dropna().astype(str).apply(len)
        fig, ax = plt.subplots(figsize=(10, 5))
        lengths.hist(bins=40, ax=ax, color='mediumseagreen', edgecolor='black')
        ax.set_title(f"{DATASET_NAME}: Text Length Distribution - {col}")
        ax.set_xlabel("Character Length")
        save_plot(fig, f"07_textlen_{col.replace('/', '_')}.png")
        print(f"\n⚠ INSIGHT: '{col}' avg length = {lengths.mean():.0f} chars, "
              f"empty/very short entries = {(lengths < 10).sum()}")

    return cat_cols


# =========================================================
# 6. TIME-BASED ANALYSIS
# =========================================================
def time_analysis(df, time_col):
    print(f"\n{'='*70}\nTIME-BASED ANALYSIS\n{'='*70}")
    if time_col not in df.columns:
        print(f"Time column '{time_col}' not found — skipping.")
        return

    dates = pd.to_datetime(df[time_col], errors='coerce')
    print(f"Date range: {dates.min()} to {dates.max()}")
    print(f"Unparseable dates: {dates.isnull().sum()}")

    # Records per year/month
    valid_dates = dates.dropna()
    if len(valid_dates) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        valid_dates.dt.to_period('M').value_counts().sort_index().plot(kind='line', ax=ax,
                                                                          marker='o', color='darkorange')
        ax.set_title(f"{DATASET_NAME}: Records Over Time")
        ax.set_ylabel("Count")
        ax.set_xlabel("Month")
        plt.xticks(rotation=45)
        save_plot(fig, "08_records_over_time.png")


# =========================================================
# 7. DATASET-SPECIFIC ANALYSIS
# =========================================================
def dataset_specific_analysis(df, dataset_type, config):
    print(f"\n{'='*70}\nDATASET-SPECIFIC ANALYSIS ({dataset_type.upper()})\n{'='*70}")

    if dataset_type == "cmapss":
        id_col, time_col = config["id_col"], config["time_col"]
        if id_col in df.columns and time_col in df.columns:
            # Cycles per engine (degradation trajectory length)
            cycles_per_unit = df.groupby(id_col)[time_col].max()
            fig, ax = plt.subplots(figsize=(10, 5))
            cycles_per_unit.hist(bins=30, ax=ax, color='teal', edgecolor='black')
            ax.set_title("Distribution of Total Cycles per Engine (Life Length)")
            ax.set_xlabel("Max Cycle (life length)")
            save_plot(fig, "09_cycles_per_engine.png")
            print(f"\n⚠ INSIGHT: Engine life ranges from {cycles_per_unit.min()} "
                  f"to {cycles_per_unit.max()} cycles (avg {cycles_per_unit.mean():.0f})")

            # Sample sensor trend for one engine (degradation pattern)
           #sensor_cols = [c for c in df.columns if c.startswith(('T', 'P', 'N')) and c not in [id_col]]
            sensor_cols = [c for c in df.columns if c.startswith(('T', 'P', 'N')) and c not in [id_col]]
            sample_id = df[id_col].iloc[0]
            sample = df[df[id_col] == sample_id]
            if sensor_cols:
                fig, axes = plt.subplots(2, 2, figsize=(14, 8))
                for ax, col in zip(axes.flat, sensor_cols[:4]):
                    ax.plot(sample[time_col], sample[col], color='crimson')
                    ax.set_title(f"{col} over cycles (Engine {sample_id})")
                    ax.set_xlabel("Cycle")
                plt.tight_layout()
                save_plot(fig, "10_sample_sensor_degradation_trend.png")

    elif dataset_type == "ngafid":
        print("Focus: maintenance frequency, time-between-maintenance, component trends")
        comp_col = next((c for c in df.columns if 'component' in c.lower()), None)
        if comp_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[comp_col].value_counts().head(15).plot(kind='barh', ax=ax, color='slateblue')
            ax.set_title("Most Frequently Maintained Components")
            ax.invert_yaxis()
            save_plot(fig, "09_top_maintained_components.png")
    
    elif dataset_type == "sdr":
        print("Focus: failure type frequency, component failure trends")
        fail_col = config.get("target_col")
        if fail_col and fail_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[fail_col].value_counts().head(15).plot(kind='barh', ax=ax, color='darkred')
            ax.set_title("Most Common Failure/Defect Types")
            ax.invert_yaxis()
            save_plot(fig, "09_top_failure_types.png")

        elif dataset_type == "sdr":
            print("Focus: failure type frequency, component failure trends")
            fail_col = config.get("target_col")
            if fail_col and fail_col in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[fail_col].value_counts().head(15).plot(kind='barh', ax=ax, color='darkred')
                ax.set_title("Most Common Failure/Defect Types (NatureOfConditionA)")
                ax.invert_yaxis()
                save_plot(fig, "09_top_failure_types.png")

            class_col = config.get("classification_col")
            if class_col and class_col in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[class_col].value_counts().head(15).plot(kind='barh', ax=ax, color='navy')
                ax.set_title("Most Common JASC (System) Codes")
                ax.invert_yaxis()
                save_plot(fig, "10_top_jasc_codes.png")

            # Component-level failure breakdown
            if "ComponentName" in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df["ComponentName"].value_counts().head(15).plot(kind='barh', ax=ax, color='darkorange')
                ax.set_title("Most Frequently Failed Components")
                ax.invert_yaxis()
                save_plot(fig, "11_top_failed_components.png")


##########################
    elif dataset_type == "ntsb":
        print("Focus: probable cause frequency, contributing factors")
        cause_col = config.get("target_col")
        if cause_col and cause_col in df.columns:
            top_causes = df[cause_col].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_causes.plot(kind='barh', ax=ax, color='goldenrod')
            ax.set_title("Most Common Probable Causes")
            ax.invert_yaxis()
            save_plot(fig, "09_top_probable_causes.png")


# =========================================================
# 8. FINAL SUMMARY REPORT
# =========================================================
def final_summary(df):
    print(f"\n{'='*70}\nFINAL EDA SUMMARY - {DATASET_NAME}\n{'='*70}")
    print(f"✓ Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"✓ Missing data columns: {df.isnull().any().sum()}")
    print(f"✓ Duplicate rows: {df.duplicated().sum()}")
    print(f"✓ Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"✓ Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"\n✓ All graphs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*70}\n")


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    config = COLUMN_CONFIG[DATASET_TYPE]

    df = load_data(FILE_PATH)
    structural_overview(df)
    missing_value_analysis(df)
    numeric_analysis(df)
    categorical_analysis(df)

    if config["time_col"]:
        time_analysis(df, config["time_col"])

    dataset_specific_analysis(df, DATASET_TYPE, config)
    final_summary(df)