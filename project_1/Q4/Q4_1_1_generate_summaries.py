#########################################################################################################################
# * Q4.1.1 - Generate Summaries
# * This script generates summaries for each patient in the dataset.
# * The summaries include statistical values and trends based on the first 48 hours of data.
# * The summaries are saved in CSV format for further analysis.
#########################################################################################################################

import pandas as pd
import numpy as np

from project_1.config import PROJ_ROOT, DATA_DIRECTORY, PROCESSED_DATA_DIR, LOGS_DIR
from project_1.loading import *

#########################################################################################################################
# ^ Data Loading
#########################################################################################################################

# Load unscaled data
df_a, df_b, df_c = load_before_scaling()
death_a, death_b, death_c = load_outcomes()

print("All data loaded from CSV.")
print(df_a.columns)


#############################################################################################################################
# ^ Data Processing
#############################################################################################################################


def generate_summary_statistical(patient_df):
    """
    generates a summary for a patient based on the most important features and includes statistical values such as min, max, mean and last value
    """
    features = {}
    features["age"] = patient_df["Age"].iloc[0]
    features["gender"] = "male" if patient_df["Gender"].iloc[0] == 1 else "female"

    summary = [
        f"Patient is a {int(features['age'])}-year-old {features['gender']}. Over the first 48 hours:"
    ]

    vital_vars = ["HCO3", "HCT", "HR", "Mg", "Na", "Temp", "BUN", "FiO2", "GCS", "K"]
    for var in vital_vars:
        if var not in patient_df.columns:
            continue
        values = patient_df[var].dropna()
        if len(values) == 0:
            continue
        min_val, max_val = values.min(), values.max()
        mean_val, last_val = values.iloc[-1], values.iloc[-1]
        summary.append(
            f"- {var} ranged from {min_val:.2f} to {max_val:.2f}, avg: {mean_val:.2f}, last: {last_val:.2f}"
        )

    return "\n".join(summary)


def get_trend_label(seq):
    """
    generates a trend label based on a three value sequence
    """
    if len(seq) < 2:
        return "unknown"
    d1, d2, d3 = (
        np.mean(seq[: len(seq) // 3]),
        np.mean(seq[len(seq) // 3 : 2 * len(seq) // 3]),
        np.mean(seq[2 * len(seq) // 3 :]),
    )
    if d1 < d2 < d3:
        return "↑"
    elif d1 > d2 > d3:
        return "↓"
    elif abs(d1 - d2) < 0.1 and abs(d2 - d3) < 0.1:
        return "→"
    else:
        return "~"


def generate_summary_trend(patient_df):
    """
    Similar to the statistical summary, but added a trend based on comparing the first 16 h with the second and last 16h of the recorded data
    """
    features = {}
    features["age"] = patient_df["Age"].iloc[0]
    features["gender"] = "male" if patient_df["Gender"].iloc[0] == 1 else "female"
    features["weight"] = patient_df["Weight"].iloc[0]

    summary = [
        f"Patient is a {int(features['age'])}-year-old {features['gender']}, weights {features['weight']} kg. Over the first 48 hours:"
    ]

    vital_vars = ["HCO3", "HCT", "HR", "Mg", "Na", "Temp", "BUN", "FiO2", "GCS", "K"]

    for var in vital_vars:
        if var not in patient_df.columns:
            continue
        values = patient_df[var].dropna()
        if len(values) == 0:
            continue
        min_val, max_val = values.min(), values.max()
        mean_val, last_val = values.iloc[-1], values.iloc[-1]
        # summary.append(f"- {var} ranged from {min_val:.2f} to {max_val:.2f}, avg: {mean_val:.2f}, last: {last_val:.2f}")
        series = patient_df[var].dropna().values
        if len(series) < 3:
            continue
        trend = get_trend_label(series)
        summary.append(
            f"- {var} ranged from {min_val:.2f} to {max_val:.2f}, shows a trend of {trend}. avg: {mean_val:.2f}, last: {last_val:.2f}"
        )

    return "\n".join(summary)


###############################################################################################################################
# ^ Generate summaries and save to CSV
###############################################################################################################################


def generate_patient_summaries(df):
    result_rows = []

    for rid, group in df.groupby("RecordID"):
        summary_stat = generate_summary_statistical(group)
        summary_trend = generate_summary_trend(group)

        result_rows.append(
            {
                "RecordID": rid,
                "summary_statistical": summary_stat,
                "summary_trend": summary_trend,
            }
        )

    return pd.DataFrame(result_rows)


summaries_a = generate_patient_summaries(df_a)
summaries_a = summaries_a.merge(death_a, on="RecordID", how="left")

summaries_b = generate_patient_summaries(df_b)
summaries_b = summaries_b.merge(death_b, on="RecordID", how="left")

summaries_c = generate_patient_summaries(df_c)
summaries_c = summaries_c.merge(death_c, on="RecordID", how="left")

print(summaries_a.loc[0, "summary_statistical"])
print()
print(summaries_a.loc[0, "summary_trend"])

# Save the summaries to CSV files
summaries_a.to_csv(PROCESSED_DATA_DIR / "set_a" / "summaries_a.csv", index=False)
summaries_b.to_csv(PROCESSED_DATA_DIR / "set_b" / "summaries_b.csv", index=False)
summaries_c.to_csv(PROCESSED_DATA_DIR / "set_c" / "summaries_c.csv", index=False)
