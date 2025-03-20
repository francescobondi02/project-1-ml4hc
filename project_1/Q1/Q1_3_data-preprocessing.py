#########################################################################################################################
#? Question 1.3: Data Preprocessing
#########################################################################################################################

import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
import logging
import numpy as np

from project_1.config import PROJ_ROOT, DATA_DIRECTORY, PROCESSED_DATA_DIR, LOGS_DIR

tqdm.pandas()

log_file_path = LOGS_DIR / "Q1_3_data-preprocessing.log"
logging.basicConfig(
    filename=str(log_file_path),
    filemode='w',  # overwrite each run; change to 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

#########################################################################################################################
# Data Loading
#########################################################################################################################

sets_dict = {}
sets = ["a", "b", "c"]

for set_name in sets:
    directory = PROCESSED_DATA_DIR / f"set_{set_name}.parquet"
    temp_set = pd.read_parquet(directory)
    sets_dict[f"set_{set_name}"] = temp_set

# Assure the loading was correct
print(sets_dict["set_a"].shape)
logging.info(f"Set A shape [start]: {sets_dict['set_a'].shape}")
print(sets_dict["set_b"].shape)
logging.info(f"Set B shape [start]: {sets_dict['set_b'].shape}")
print(sets_dict["set_c"].shape)
logging.info(f"Set C shape [start]: {sets_dict['set_c'].shape}")

#########################################################################################################################
# Outlier Removal
#########################################################################################################################

def clean_df(df):
    """
    Clean the input DataFrame according to the following rules:
      - Missing value handling: For Age, Gender, Height, ICUType, Weight, set -1 to NA.
      - Height outlier removal: Set Height to NA if < 100 cm or >= 300 cm.
      - Weight outlier removal: Set Weight to NA if < 20 kg or >= 300 kg.
      - PaO2 corrections: Set PaO2 equal to 0 to NA; if PaO2 equals 7.47, correct it to 74.7.
      - pH unit correction: If pH is between 65 and 80, divide by 10; if between 650 and 800, divide by 100.
      - Temperature corrections: Set Temp to NA if Temp is less than 20.
    """
    # 1. Missing value handling: Replace -1 with np.nan for selected columns.
    missing_cols = ["Age", "Gender", "Height", "ICUType", "Weight"]
    for col in missing_cols:
        df.loc[df[col] == -1, col] = np.nan

    # 2. Height outlier removal: Set Height to NA if < 100 or >= 300.
    df.loc[(df["Height"] < 100) | (df["Height"] >= 300), "Height"] = np.nan

    # 3. Weight outlier removal: Set Weight to NA if < 20 or >= 300.
    df.loc[(df["Weight"] < 20) | (df["Weight"] >= 300), "Weight"] = np.nan

    # 4. PaO2 corrections:
    #    Set PaO2 equal to 0 to NA, and if PaO2 is 7.47, correct it to 74.7.
    df.loc[df["PaO2"] == 0, "PaO2"] = np.nan
    df.loc[df["PaO2"] == 7.47, "PaO2"] = 74.7

    # 5. pH unit correction:
    #    If pH is between 65 and 80, divide by 10; if between 650 and 800, divide by 100.
    def correct_ph(ph):
        if pd.isna(ph):
            return ph
        if 65 <= ph <= 80:
            return ph / 10.0
        elif 650 <= ph <= 800:
            return ph / 100.0
        else:
            return ph
    df["pH"] = df["pH"].apply(correct_ph)

    # 6. Temperature corrections: Set Temp to NA if Temp is < 20.
    df.loc[df["Temp"] < 20, "Temp"] = np.nan

    return df

for set_key, df in sets_dict.items():
    # Clean the DataFrame
    cleaned_df = clean_df(df)
    sets_dict[set_key] = cleaned_df  # Update dictionary (optional)

    # Export to Parquet file (e.g., "set_a_cleaned.parquet")
    output_filename = PROCESSED_DATA_DIR / f"{set_key}_cleaned.parquet"
    cleaned_df.to_parquet(output_filename, index=False)
    print(f"Cleaned data for {set_key} saved as {output_filename}")
    logging.info(f"Cleaned data for {set_key} saved as {output_filename}")

#########################################################################################################################
# Forward Filling 
#########################################################################################################################

def forward_fill(df):
    # Ensure the DataFrame is sorted by RecordID and Time
    df.sort_values(by=["RecordID", "Time"], inplace=True)

    # Get a list of all columns except "RecordID" and "Time"
    other_cols = [col for col in df.columns if col != "RecordID" and col != "Time"]

    # Group by RecordID and apply forward fill for each group.
    df[other_cols] = df.groupby("RecordID")[other_cols].ffill()

    return df

for set_key, df in sets_dict.items():
    # Forward fill the DataFrame
    filled_df = forward_fill(df)
    sets_dict[set_key] = filled_df  # Update dictionary (optional)

#########################################################################################################################
# Interpolation as Backward Filling
#########################################################################################################################

def time_based_interpolation(df):
    """
    Perform time-based interpolation on the DataFrame.
    
    This function:
      - Converts the "Time" column to datetime,
      - Sets "Time" as the index,
      - Interpolates numeric columns (excluding "RecordID") using method='time'
        with limit_direction='both',
      - Resets the index to restore "Time" as a regular column.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with at least "Time" and "RecordID" columns.
    
    Returns:
      pd.DataFrame: The DataFrame with interpolated values.
    """
    # Ensure the "Time" column is in datetime format.
    df["Time"] = pd.to_datetime(df["Time"])
    
    # Set "Time" as the DataFrame index for time-based interpolation.
    df = df.set_index("Time")
    
    # Identify the columns to interpolate (exclude non-numeric columns like "RecordID").
    cols_to_interp = [col for col in df.columns if col != "RecordID"]
    
    # Apply time-based interpolation; limit_direction='both' fills NaNs at the start and end too.
    df[cols_to_interp] = df[cols_to_interp].interpolate(method='time', limit_direction='both')
    
    # Restore "Time" as a regular column by resetting the index.
    df = df.reset_index()
    
    return df

for key, df in sets_dict.items():
    sets_dict[key] = time_based_interpolation(df)
    print(f"After interpolation, {key} has shape: {sets_dict[key].shape}")
    logging.info(f"After interpolation, {key} has shape: {sets_dict[key].shape}")

# Saving temporary data for plotting 
for set_name, set_df in tqdm(sets_dict.items(), desc="Storing DataFrames", unit="set"):
    output_path = PROCESSED_DATA_DIR / f"{set_name}_to_scale.parquet"
    logging.info(f"{set_name} final shape: {set_df.shape}")
    set_df.to_parquet(output_path, index=False, engine = "pyarrow")
    print(f"Saved {output_path}")
    logging.info(f"Saved {output_path}")

#########################################################################################################################
# Scale the Data
#########################################################################################################################

from sklearn.preprocessing import StandardScaler, RobustScaler

cols_to_scale = [col for col in df.columns if col not in ["RecordID", "Time", "Gender"]]

### For normally-distributed columns, we use the StandardScaler
### For non-normally-distributed columns, we use the RobustScaler

nd_cols = ["Height", "Weight", "Age", "Albumin", "Cholesterol", "DiasABP", "HCO3", "HCT", "HR", "Mg", "MAP", "Na", "NIDiasABP", "NIMAP", "NISysABP", "SysABP", "PaCO2", "PaO2", "Platelets", "RespRate", "Temp"]
nnd_cols = [col for col in cols_to_scale if col not in nd_cols]

scaler_nd = StandardScaler()
scaler_nnd = RobustScaler()

# Process each set: fit on set_a, then only transform on the others.
for set_key in ["set_a", "set_b", "set_c"]:
    df = sets_dict[set_key]
    if set_key == "set_a":
        # Fit on the first set
        scaled_values_nd = scaler_nd.fit_transform(df[nd_cols])
        scaled_values_nnd = scaler_nnd.fit_transform(df[nnd_cols])
    else:
        # Transform the other sets using the fitted scalers
        scaled_values_nd = scaler_nd.transform(df[nd_cols])
        scaled_values_nnd = scaler_nnd.transform(df[nnd_cols])
    
    # Convert the scaled numpy arrays to DataFrames while preserving the index
    df_scaled_nd = pd.DataFrame(scaled_values_nd, columns=nd_cols, index=df.index)
    df_scaled_nnd = pd.DataFrame(scaled_values_nnd, columns=nnd_cols, index=df.index)
    
    # Combine the scaled DataFrames along the columns axis
    df_scaled = pd.concat([df_scaled_nd, df_scaled_nnd], axis=1)
    
    # Combine the unmodified columns with the scaled columns.
    df_final = pd.concat([df[["RecordID", "Time", "Gender"]].reset_index(drop=True),
                          df_scaled.reset_index(drop=True)], axis=1)
    
    # Update the dictionary with the final DataFrame
    sets_dict[set_key] = df_final

# Optionally, print the first 10 rows of set_a to check the result.
print(sets_dict["set_a"].head(10))

#########################################################################################################################
# Save the Processed Data
#########################################################################################################################

for set_name, set_df in tqdm(sets_dict.items(), desc="Storing DataFrames", unit="set"):
    output_path = PROCESSED_DATA_DIR / f"{set_name}_final.parquet"
    logging.info(f"{set_name} final shape: {set_df.shape}")
    set_df.to_parquet(output_path, index=False, engine = "pyarrow")
    print(f"Saved {output_path}")
    logging.info(f"Saved {output_path}")

print("\nAll DataFrames have been saved to Parquet format.")
logging.info("All DataFrames have been saved to Parquet format.")