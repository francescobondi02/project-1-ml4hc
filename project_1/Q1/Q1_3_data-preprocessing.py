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
from project_1.features import fill_missing_times

tqdm.pandas()

SEED = 42
np.random.seed(SEED)

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
    directory = PROCESSED_DATA_DIR / f"set_{set_name}" / f"set_{set_name}.parquet"
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
# Check that -1 values in the static variables only appear in the first row for each patient
#########################################################################################################################

# Assuming 'data' is your DataFrame

def check_neg_vals(df):
    # Group by RecordID and check for -1
    def check_first_row(group):
        return pd.Series({
            'Age_First_Only': (group['Age'].iloc[1:] != -1).all(),
            'Gender_First_Only': (group['Gender'].iloc[1:] != -1).all(),
            'Height_First_Only': (group['Height'].iloc[1:] != -1).all(),
            'Weight_First_Only': (group['Weight'].iloc[1:] != -1).all()
        })

    # Apply the function to each group
    result = df.groupby('RecordID').apply(check_first_row).reset_index()

    # Print the result
    #print(result)

    # Replace -1 with NA in the specified columns
    df.replace({
        'Height': {-1: None},
        'Age': {-1: None},
        'Weight': {-1: None},
        'ICUType': {-1: None},
        'Gender': {-1: None}
    }, inplace=True)

    # Check if any RecordID has -1 in non-first rows
    violations = result[
        (~result['Age_First_Only']) |
        (~result['Gender_First_Only']) |
        (~result['Height_First_Only']) |
        (~result['Weight_First_Only'])
    ]

    # Print violations
    #print(violations)
    if violations.empty:
        logging.info("No violations found.")
        logging.info("\n")

for set_key, df in sets_dict.items():
    check_neg_vals(df)

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
    output_filename = PROCESSED_DATA_DIR / f"{set_key}" / f"{set_key}_before_imputation.parquet"
    cleaned_df.to_parquet(output_filename, index=False)
    print(f"Cleaned data for {set_key} saved as {output_filename}")
    logging.info(f"Cleaned data for {set_key} saved as {output_filename}")

#########################################################################################################################
# Imputation for Missing Static Features
#########################################################################################################################

from sklearn.impute import KNNImputer
def knn_impute_static_features(df, static_features=["Age", "Weight", "Height", "Gender"], n_neighbors=10):
    """
    Impute missing static values (currently indicated by -1) using KNN imputation with n_neighbors.
    
    Parameters:
      df (pd.DataFrame): DataFrame with one row per patient.
      static_features (list): List of static feature column names to impute.
      n_neighbors (int): Number of neighbors to use for KNN imputation.
      
    Returns:
      pd.DataFrame: The DataFrame with missing static feature values imputed.
    """
    # Work on a copy to avoid modifying the original DataFrame.
    df_impute = df.copy()
    
    # Replace missing values (-1) with np.nan in the static columns.
    df_impute[static_features] = df_impute[static_features].replace(-1, np.nan)
    
    # Initialize the KNN imputer.
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Fit and transform the static features.
    imputed_array = imputer.fit_transform(df_impute[static_features])
    
    # Create a new DataFrame with the imputed static features.
    df_imputed_static = pd.DataFrame(imputed_array, columns=static_features, index=df_impute.index)
    
    # Update the original DataFrame with the imputed values.
    df_impute.update(df_imputed_static)
    
    return df_impute

for set_key, df in sets_dict.items():
    # Impute missing static features
    # Get the static features in df
    static_df = df.groupby("RecordID", as_index=False).first()[["RecordID", "Age", "Weight", "Height", "Gender"]].copy()
    imputed_df = knn_impute_static_features(static_df)

    if imputed_df.index.name == "RecordID":
        imputed_df = imputed_df.reset_index()

    # Check if there are any NaN values
    if imputed_df.isnull().values.any():
        logging.warning(f"NaN values found in imputed static features for {set_key}.")
        logging.info(f"NaN values found in imputed static features for {set_key}.")

    # Update the original DataFrame with the imputed values
    static_cols = ["Age", "Weight", "Height", "Gender"]
    # Create a mapping for each column and update df_full.
    for col in static_cols:
        mapping = imputed_df.set_index("RecordID")[col]
        df[col] = df["RecordID"].map(mapping)

    #sets_dict[set_key] = imputed_df  # Update dictionary (optional)

    # Reorder columns

    # Assume that the first two columns should remain in place.
    # For example, we assume these are the first two columns of the DataFrame.
    first_two = list(df.columns[:2])
    
    # The rest of the columns, excluding the static columns.
    remaining = [col for col in df.columns if col not in static_cols and col not in first_two]
    
    # Create the new order: first two columns, then the static columns, then the remaining columns.
    new_order = first_two + static_cols + remaining
    
    # Reorder the DataFrame and return
    df = df[new_order]

    # Check if new dataframe has NaN values on the static columns
    if df[static_cols].isnull().values.any():
        logging.warning(f"NaN values found in imputed static features for {set_key}.")
        logging.info(f"NaN values found in imputed static features for {set_key}.")
    logging.info(f"{df.head(10)}")
    print(f"{df.head(10)}")

    for set_key, df in sets_dict.items():
        # Export to Parquet file (e.g., "set_a....parquet")
        output_filename = PROCESSED_DATA_DIR / f"{set_key}" / f"{set_key}_before_ffill.parquet"
        cleaned_df.to_parquet(output_filename, index=False)
        print(f"Cleaned data for {set_key} saved as {output_filename}")
        logging.info(f"Cleaned data for {set_key} saved as {output_filename}")

#########################################################################################################################
# Adding Missing Time Points
#########################################################################################################################

"""for set_key, df in sets_dict.items():
    # Fill in missing time points
    filled_df = df.groupby("RecordID", group_keys=False).apply(fill_missing_times)
    sets_dict[set_key] = filled_df  # Update dictionary (optional)

print("Shapes of DataFrames after adding missing timesteps:")
logging.info("Shapes of DataFrames after adding missing timesteps:")
for set_key, df in sets_dict.items():
    print(f"{set_key}: {df.shape}")
    logging.info(f"{set_key}: {df.shape}")"""

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

    # Export to Parquet file (e.g., "set_a....parquet")
    output_filename = PROCESSED_DATA_DIR / f"{set_key}" / f"{set_key}_before_backward.parquet"
    filled_df.to_parquet(output_filename, index=False)
    print(f"Forward-filled data for {set_key} saved as {output_filename}")
    logging.info(f"Forward-filled data for {set_key} saved as {output_filename}")

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
    cols_to_interp = [col for col in df.columns if col not in ["RecordID", "Time"]]
    
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
    output_path = PROCESSED_DATA_DIR / f"{set_name}" / f"{set_name}_before_scaling.parquet"
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
    output_path = PROCESSED_DATA_DIR / f"{set_name}" / f"{set_name}_final.parquet"
    logging.info(f"{set_name} final shape: {set_df.shape}")
    set_df.to_parquet(output_path, index=False, engine = "pyarrow")
    print(f"Saved {output_path}")
    logging.info(f"Saved {output_path}")

print("\nAll DataFrames have been saved to Parquet format.")
logging.info("All DataFrames have been saved to Parquet format.")