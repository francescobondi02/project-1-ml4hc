#########################################################################################################################
# * Q1.1: Data Transformation
# * This script processes the raw data files from the official dataset folder (located under data/data_1/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0)
# * and transforms them into a usable dataframe format. The dfs are saved in parquet format for efficient storage and retrieval.
#########################################################################################################################

import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path

from project_1.config import PROJ_ROOT, DATA_DIRECTORY, PROCESSED_DATA_DIR, LOGS_DIR

tqdm.pandas()

#########################################################################################################################
# ^ Data Loading
#########################################################################################################################

# Load the data (for the first time)
sets = ["a", "b", "c"]
sets_dict = {}
for set_name in sets:
    set_folder = DATA_DIRECTORY / f"set-{set_name}"
    print(f"Reading data from {set_folder}")

    patient_data_list = []

    files = [f for f in os.listdir(set_folder) if f.endswith(".txt")]
    for filename in tqdm(
        files, desc=f"Processing files in set-{set_name}", unit="file"
    ):
        file_path = os.path.join(set_folder, filename)

        # Read patient file
        patient_df = pd.read_csv(file_path)

        # Extract RecordID from the 'Parameter' column where value is 'RecordID'
        record_id = patient_df.loc[
            patient_df["Parameter"] == "RecordID", "Value"
        ].values[0]

        # Pivot to transform measurements, using 'first' to resolve duplicates
        patient_df = patient_df.pivot_table(
            index="Time", columns="Parameter", values="Value", aggfunc="first"
        )
        patient_df.reset_index(inplace=True)

        # Remove any existing RecordID column and insert our extracted one as the first column
        if "RecordID" in patient_df.columns:
            patient_df.drop(columns=["RecordID"], inplace=True)
        patient_df.insert(0, "RecordID", record_id)

        # Append the processed DataFrame to the list
        patient_data_list.append(patient_df)

    # Combine all patient data into a single DataFrame
    patients_df = pd.concat(patient_data_list, ignore_index=True)

    # Store in dictionary
    sets_dict["set_" + set_name] = patients_df

# Output the first 5 rows of the Set A DataFrame
print(sets_dict["set_a"].head())
print(sets_dict["set_a"].shape)

##########################################################################################
# ^ Creating Data Processed Directory
##########################################################################################

# Create the directory if it doesn't exist
set_a_path = Path(PROCESSED_DATA_DIR / "set_a")
set_b_path = Path(PROCESSED_DATA_DIR / "set_b")
set_c_path = Path(PROCESSED_DATA_DIR / "set_c")

set_a_path.mkdir(parents=True, exist_ok=True)
set_b_path.mkdir(parents=True, exist_ok=True)
set_c_path.mkdir(parents=True, exist_ok=True)

##########################################################################################
# ^ Discretizing Time Column
##########################################################################################

base_date = "2025-03-10"  # Format is YYYY-MM-DD (date is chosen randomly)


# Function to fix invalid times
def adjust_time(time_str, base_date):
    # Split hours and minutes
    hours, minutes = map(int, time_str.split(":"))

    # Calculate valid hour & days overflow
    day_offset = hours // 24  # Number of days to add
    new_hour = hours % 24  # Wrapped hour (0-23)

    # Create the corrected datetime
    corrected_datetime = datetime.strptime(base_date, "%Y-%m-%d") + timedelta(
        days=day_offset, hours=new_hour, minutes=minutes
    )

    return corrected_datetime


def round_up_next_hour(ts):
    # If timestamp is exactly on the hour, return it unchanged.
    if ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        return ts
    # Otherwise, round up to the next hour.
    return ts.ceil("H")


# Apply the functions to the 'Time' column (for each sets)
for set_name, set_df in tqdm(sets_dict.items(), desc="Discretizing time", unit="set"):
    # Convert 'Time' column from string to datetime using the adjust_time function.
    set_df["Time"] = set_df["Time"].progress_apply(lambda x: adjust_time(x, base_date))
    # Round up each timestamp to the next hour, except if it is exactly on the hour.
    set_df["Time"] = set_df["Time"].progress_apply(round_up_next_hour)
    # Group by RecordID and discretized Time, taking the mean in case of multiple measurements.
    sets_dict[set_name] = set_df.groupby(["RecordID", "Time"], as_index=False).mean()

# Output the first 5 rows of the Set A DataFrame
print(sets_dict["set_a"].head())

##########################################################################################
# ^ Store Dataframes to Parquet
##########################################################################################

for set_name, set_df in tqdm(sets_dict.items(), desc="Storing DataFrames", unit="set"):
    output_path = PROCESSED_DATA_DIR / f"{set_name}" / f"{set_name}.parquet"
    set_df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved {output_path}")

print("\nAll DataFrames have been saved to Parquet format.")
