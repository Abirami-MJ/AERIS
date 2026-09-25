"""
AERIS NGAFID Feature Extractor (Phase 1)
Implements flight cleaning, relative 5-segment temporal partitioning,
and exact 437-feature extraction.
"""

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "volt1", "volt2", "amp1", "amp2",
    "FQtyL", "FQtyR",
    "E1 FFlow", "E1 OilT", "E1 OilP", "E1 RPM",
    "E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4",
    "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4",
    "OAT", "IAS", "VSpd", "NormAc", "AltMSL"
]

UNTOUCHED_SENSORS = {
    "volt2", "amp2", "E1 CHT2", "E1 CHT3", "E1 CHT4", "NormAc", "AltMSL"
}

def clean_flight_dataframe(flight_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Cleans a single flight DataFrame:
    1. Sorts chronologically by timestep.
    2. Preserves original raw missingness mask before interpolation.
    3. Leaves untouched sensors un-imputed.
    4. Safe sensors: linear interpolation with limit=5, limit_area="inside".
    """
    # 1. Sort chronologically
    df_sorted = flight_df.sort_values(by="timestep").copy()
    
    # 2. Preserve raw missingness masks
    raw_missing_masks = {}
    for col in SENSOR_COLUMNS:
        if col in df_sorted.columns:
            raw_missing_masks[col] = df_sorted[col].isna().to_numpy()
        else:
            raw_missing_masks[col] = np.ones(len(df_sorted), dtype=bool)

    # 3 & 4. Clean safe sensors, leave untouched untouched
    for col in SENSOR_COLUMNS:
        if col in df_sorted.columns:
            if col not in UNTOUCHED_SENSORS:
                # Linear interpolation, max gap = 5, inside only, no boundary extrapolation
                df_sorted[col] = df_sorted[col].interpolate(
                    method="linear", limit=5, limit_area="inside"
                )
    
    return df_sorted, raw_missing_masks

def extract_features_single_flight(flight_df: pd.DataFrame, master_index: int = None) -> dict:
    """
    Extracts exactly 437 features for a single flight DataFrame.
    """
    n_rows = len(flight_df)
    if n_rows == 0:
        raise ValueError("Cannot extract features from an empty flight.")
    
    # Clean flight
    cleaned_df, raw_missing_masks = clean_flight_dataframe(flight_df)
    
    # Timestep vector for slope calculation
    timesteps = cleaned_df["timestep"].to_numpy().astype(float)
    
    # 5 relative temporal segments: relative to flight length
    # Uses monotonic integer division: (arange(n_rows) * 5) // n_rows (0, 1, 2, 3, 4)
    seg_assignments = (np.arange(n_rows) * 5) // n_rows
    
    features = {}
    if master_index is not None:
        features["Master Index"] = master_index
    
    for sensor in SENSOR_COLUMNS:
        if sensor not in cleaned_df.columns:
            vals = np.full(n_rows, np.nan)
            raw_mask = np.ones(n_rows, dtype=bool)
        else:
            vals = cleaned_df[sensor].to_numpy().astype(float)
            raw_mask = raw_missing_masks[sensor]
        
        # 1. Segment statistics (5 segments x 3 stats = 15 features per sensor)
        for seg_idx in range(5):
            seg_num = seg_idx + 1
            in_seg = (seg_assignments == seg_idx)
            seg_vals = vals[in_seg]
            seg_raw_mask = raw_mask[in_seg]
            
            # pct_missing from ORIGINAL raw mask
            pct_missing = float(np.mean(seg_raw_mask)) if len(seg_raw_mask) > 0 else np.nan
            
            # valid values in segment after cleaning
            valid_seg = seg_vals[~np.isnan(seg_vals)]
            if len(valid_seg) == 0:
                mean_val = np.nan
                std_val = np.nan
            elif len(valid_seg) == 1:
                mean_val = float(valid_seg[0])
                std_val = np.nan  # Sample std requires >= 2 points
            else:
                mean_val = float(np.mean(valid_seg))
                std_val = float(np.std(valid_seg, ddof=1))
            
            features[f"{sensor}_seg{seg_num}_mean"] = mean_val
            features[f"{sensor}_seg{seg_num}_std"] = std_val
            features[f"{sensor}_seg{seg_num}_pct_missing"] = pct_missing
        
        # 2. Whole-flight statistics (4 features per sensor)
        valid_idx = np.where(~np.isnan(vals))[0]
        n_valid = int(len(valid_idx))
        
        if n_valid < 2:
            slope = np.nan
            first_last_change = np.nan
            first_last_pct_change = np.nan
        else:
            first_idx = valid_idx[0]
            last_idx = valid_idx[-1]
            first_val = float(vals[first_idx])
            last_val = float(vals[last_idx])
            
            # Slope: OLS slope of valid values vs timestep
            t_valid = timesteps[valid_idx]
            y_valid = vals[valid_idx]
            t_mean = np.mean(t_valid)
            y_mean = np.mean(y_valid)
            denom = np.sum((t_valid - t_mean) ** 2)
            if denom == 0:
                slope = np.nan
            else:
                numer = np.sum((t_valid - t_mean) * (y_valid - y_mean))
                slope = float(numer / denom)
            
            # Changes
            first_last_change = float(last_val - first_val)
            if first_val == 0.0:
                first_last_pct_change = np.nan
            else:
                first_last_pct_change = float((last_val - first_val) / abs(first_val))
        
        features[f"{sensor}_slope"] = slope
        features[f"{sensor}_first_last_change"] = first_last_change
        features[f"{sensor}_first_last_pct_change"] = first_last_pct_change
        features[f"{sensor}_n_valid"] = n_valid

    return features
