"""
AERIS - C-MAPSS Data Loader
================================================
NASA C-MAPSS files (train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt) have
NO header row and are SPACE-separated (with trailing extra spaces causing
2 empty columns at the end - this is a known quirk of the raw files).

This script loads them into proper labeled DataFrames.
"""

import pandas as pd

# =========================================================
# CONFIG - change which sub-dataset you're loading (FD001-FD004)
# =========================================================
FD_NUMBER = "FD001"          # 'FD001' | 'FD002' | 'FD003' | 'FD004'
DATA_FOLDER = "."            # folder where your .txt files are

# Official column names
COLUMN_NAMES = [
    "unit_number", "time_cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "T2", "T24", "T30", "T50", "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc",
    "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"
]


def load_cmapss_file(filepath, columns, has_rul=False):
    """
    Loads a raw C-MAPSS .txt file (space-separated, no header).
    Handles the trailing-whitespace quirk that creates extra empty columns.
    """
    if has_rul:
        # RUL files have just 1 column
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=["RUL"])
    else:
        df = pd.read_csv(filepath, sep=r'\s+', header=None)
        # Raw files sometimes parse with 2 extra trailing NaN columns - trim to 26
        df = df.iloc[:, :len(columns)]
        df.columns = columns
    return df


if __name__ == "__main__":
    train_path = f"{DATA_FOLDER}/train_{FD_NUMBER}.txt"
    test_path = f"{DATA_FOLDER}/test_{FD_NUMBER}.txt"
    rul_path = f"{DATA_FOLDER}/RUL_{FD_NUMBER}.txt"

    train_df = load_cmapss_file(train_path, COLUMN_NAMES)
    test_df = load_cmapss_file(test_path, COLUMN_NAMES)
    rul_df = load_cmapss_file(rul_path, COLUMN_NAMES, has_rul=True)

    print(f"Train shape: {train_df.shape}")
    print(train_df.head())
    print(f"\nTest shape: {test_df.shape}")
    print(f"\nRUL shape: {rul_df.shape}")

    # Save as clean CSVs for reuse in your EDA/cleaning scripts
    train_df.to_csv(f"train_{FD_NUMBER}_labeled.csv", index=False)
    test_df.to_csv(f"test_{FD_NUMBER}_labeled.csv", index=False)
    rul_df.to_csv(f"RUL_{FD_NUMBER}_labeled.csv", index=False)
    print(f"\n✓ Saved labeled CSVs: train_{FD_NUMBER}_labeled.csv, "
          f"test_{FD_NUMBER}_labeled.csv, RUL_{FD_NUMBER}_labeled.csv")