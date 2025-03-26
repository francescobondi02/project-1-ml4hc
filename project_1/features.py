import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

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

def scale_features_advanced(set_a, set_b, set_c):
    """
        Scale the features of the dataset, after having studied the features
    """

    cols_to_scale = [col for col in set_a.columns if col not in ["RecordID", "Time", "Gender"]]

    ### For normally-distributed columns, we use the StandardScaler
    ### For non-normally-distributed columns, we use the RobustScaler

    nd_cols = ["Height", "Weight", "Age", "Albumin", "Cholesterol", "DiasABP", "HCO3", "HCT", "HR", "Mg", "MAP", "Na", "NIDiasABP", "NIMAP", "NISysABP", "SysABP", "PaCO2", "PaO2", "Platelets", "RespRate", "Temp"]
    nnd_cols = [col for col in cols_to_scale if col not in nd_cols]

    scaler_nd = StandardScaler()
    scaler_nnd = RobustScaler()

    # Process each set: fit on set_a, then only transform on the others.
    # Fit on the first set
    scaled_values_nd_a = scaler_nd.fit_transform(set_a[nd_cols])
    scaled_values_nnd_a = scaler_nnd.fit_transform(set_a[nnd_cols])
    # Transform the other sets using the fitted scalers
    scaled_values_nd_b = scaler_nd.transform(set_b[nd_cols])
    scaled_values_nnd_b = scaler_nnd.transform(set_b[nnd_cols])

    scaled_values_nd_c = scaler_nd.transform(set_c[nd_cols])
    scaled_values_nnd_c = scaler_nnd.transform(set_c[nnd_cols])
        
    # Convert the scaled numpy arrays to DataFrames while preserving the index
    df_scaled_nd_a = pd.DataFrame(scaled_values_nd_a, columns=nd_cols, index=set_a.index)
    df_scaled_nnd_a = pd.DataFrame(scaled_values_nnd_a, columns=nnd_cols, index=set_a.index)

    df_scaled_nd_b = pd.DataFrame(scaled_values_nd_b, columns=nd_cols, index=set_b.index)
    df_scaled_nnd_b = pd.DataFrame(scaled_values_nnd_b, columns=nnd_cols, index=set_b.index)

    df_scaled_nd_c = pd.DataFrame(scaled_values_nd_c, columns=nd_cols, index=set_c.index)
    df_scaled_nnd_c = pd.DataFrame(scaled_values_nnd_c, columns=nnd_cols, index=set_c.index)
    
    # Combine the scaled DataFrames along the columns axis
    df_scaled_a = pd.concat([df_scaled_nd_a, df_scaled_nnd_a], axis=1)
    df_scaled_b = pd.concat([df_scaled_nd_b, df_scaled_nnd_b], axis=1)
    df_scaled_c = pd.concat([df_scaled_nd_c, df_scaled_nnd_c], axis=1)
    
    # Combine the unmodified columns with the scaled columns.
    df_final_a = pd.concat([set_a[["RecordID", "Time", "Gender"]].reset_index(drop=True),
                        df_scaled_a.reset_index(drop=True)], axis=1)
    
    df_final_b = pd.concat([set_b[["RecordID", "Time", "Gender"]].reset_index(drop=True),
                        df_scaled_b.reset_index(drop=True)], axis=1)
    
    df_final_c = pd.concat([set_c[["RecordID", "Time", "Gender"]].reset_index(drop=True),
                        df_scaled_c.reset_index(drop=True)], axis=1)
    return df_final_a, df_final_b, df_final_c

def scale_features_basic(set_a, set_b, set_c):
    """
    Scale the features of the dataset using MinMax scaling for every feature.
    The scaler is fit on set_a and then used to transform set_b and set_c.
    """
    # Define the columns to scale (exclude RecordID, Time, and Gender)
    cols_to_scale = [col for col in set_a.columns if col not in ["RecordID", "Time", "Gender"]]
    
    # Initialize and fit the MinMaxScaler on set_a's features
    scaler = MinMaxScaler()
    scaled_values_a = scaler.fit_transform(set_a[cols_to_scale])
    
    # Transform the features of set_b and set_c
    scaled_values_b = scaler.transform(set_b[cols_to_scale])
    scaled_values_c = scaler.transform(set_c[cols_to_scale])
    
    # Convert the scaled arrays back to DataFrames with the same columns and indices
    df_scaled_a = pd.DataFrame(scaled_values_a, columns=cols_to_scale, index=set_a.index)
    df_scaled_b = pd.DataFrame(scaled_values_b, columns=cols_to_scale, index=set_b.index)
    df_scaled_c = pd.DataFrame(scaled_values_c, columns=cols_to_scale, index=set_c.index)
    
    # Combine the unmodified columns with the scaled columns
    df_final_a = pd.concat([set_a[["RecordID", "Time", "Gender"]].reset_index(drop=True),
                            df_scaled_a.reset_index(drop=True)], axis=1)
    df_final_b = pd.concat([set_b[["RecordID", "Time", "Gender"]].reset_index(drop=True),
                            df_scaled_b.reset_index(drop=True)], axis=1)
    df_final_c = pd.concat([set_c[["RecordID", "Time", "Gender"]].reset_index(drop=True),
                            df_scaled_c.reset_index(drop=True)], axis=1)
    
    return df_final_a, df_final_b, df_final_c


__all__ = ['fill_missing_times', "scale_features_advanced", "scale_features_basic"]