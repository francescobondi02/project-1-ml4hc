import numpy as np
import pandas as pd
from tqdm import tqdm

def fill_missing_times(group):
    # Ensure the "Time" column is a datetime object
    group['Time'] = pd.to_datetime(group['Time'])
    # Sort the group by time
    group = group.sort_values('Time').reset_index(drop=True)
    
    new_rows = []
    
    # Fill gaps between consecutive measurements
    for i in range(len(group) - 1):
        current_time = group.loc[i, 'Time']
        next_time = group.loc[i + 1, 'Time']
        diff_hours = (next_time - current_time).total_seconds() / 3600.0
        
        if diff_hours >= 2:  # If there's a gap of at least 2 hours, add missing hours
            missing_times = pd.date_range(
                start=current_time + pd.Timedelta(hours=1), 
                end=next_time - pd.Timedelta(hours=1),
                freq='h'
            )
            for missing_time in missing_times:
                new_row = group.loc[i].copy()  # use current row as a template
                new_row['Time'] = missing_time
                # Set measurement columns to NaN (all columns except RecordID and Time)
                for col in group.columns:
                    if col not in ['RecordID', 'Time']:
                        new_row[col] = np.nan
                new_rows.append(new_row)
    
    # Determine the expected final time: 48 hours after the first measurement
    expected_end_time = group.loc[0, 'Time'] + pd.Timedelta(hours=48)
    last_time = group.loc[len(group) - 1, 'Time']
    diff_hours_end = (expected_end_time - last_time).total_seconds() / 3600.0
    
    # If the gap at the end is at least 1 hour, fill in missing times until expected_end_time
    if diff_hours_end >= 1:
        missing_times = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            end=expected_end_time,
            freq='h'
        )
        for missing_time in missing_times:
            new_row = group.iloc[-1].copy()  # use the last row as a template
            new_row['Time'] = missing_time
            for col in group.columns:
                if col not in ['RecordID', 'Time']:
                    new_row[col] = np.nan
            new_rows.append(new_row)
    
    # If there are new rows, add them and sort by time
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        group = pd.concat([group, new_df], ignore_index=True)
    
    group = group.sort_values('Time').reset_index(drop=True)
    return group

def scale_features_advanced(df):
    """
        Scale the features of the dataset, after having studied the features
    """
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

__all__ = ['fill_missing_times']