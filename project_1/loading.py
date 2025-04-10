import pandas as pd

from project_1.config import PROCESSED_DATA_DIR, PROJ_ROOT

# Loading function for different datasets

def load_initial_data():
    """
        Load the basic datasets.
        This datasets only have the full timesteps
    """
    set_a = pd.read_parquet(PROCESSED_DATA_DIR / "set_a" / "set_a.parquet")
    set_b = pd.read_parquet(PROCESSED_DATA_DIR / "set_b" / "set_b.parquet")
    set_c = pd.read_parquet(PROCESSED_DATA_DIR / "set_c" / "set_c.parquet")
    print("Shapes of the datasets:")
    print(f"Set A: {set_a.shape}", f"Set B: {set_b.shape}", f"Set C: {set_c.shape}")
    return set_a, set_b, set_c

def load_before_imputation():
    """
        Load the cleaned datasets.
        This datasets have the full timesteps and the cleaned data
    """
    set_a = pd.read_parquet(PROCESSED_DATA_DIR / "set_a" / "set_a_before_imputation.parquet")
    set_b = pd.read_parquet(PROCESSED_DATA_DIR / "set_b" / "set_b_before_imputation.parquet")
    set_c = pd.read_parquet(PROCESSED_DATA_DIR / "set_c" / "set_c_before_imputation.parquet")
    print("Shapes of the datasets:")
    print(f"Set A: {set_a.shape}", f"Set B: {set_b.shape}", f"Set C: {set_c.shape}")
    return set_a, set_b, set_c

def load_before_ffill():
    """
        Load the not scaled datasets.
        This datasets have the full timesteps, the cleaned data and the data not scaled
    """
    set_a = pd.read_parquet(PROCESSED_DATA_DIR / "set_a" / "set_a_before_ffill.parquet")
    set_b = pd.read_parquet(PROCESSED_DATA_DIR / "set_b" / "set_b_before_ffill.parquet")
    set_c = pd.read_parquet(PROCESSED_DATA_DIR / "set_c" / "set_c_before_ffill.parquet")
    print("Shapes of the datasets:")
    print(f"Set A: {set_a.shape}", f"Set B: {set_b.shape}", f"Set C: {set_c.shape}")
    return set_a, set_b, set_c

def load_before_backward():
    """
        Load the not scaled datasets.
        This datasets have the full timesteps, the cleaned data and the data not scaled
    """
    set_a = pd.read_parquet(PROCESSED_DATA_DIR / "set_a" / "set_a_before_backward.parquet")
    set_b = pd.read_parquet(PROCESSED_DATA_DIR / "set_b" / "set_b_before_backward.parquet")
    set_c = pd.read_parquet(PROCESSED_DATA_DIR / "set_c" / "set_c_before_backward.parquet")
    print("Shapes of the datasets:")
    print(f"Set A: {set_a.shape}", f"Set B: {set_b.shape}", f"Set C: {set_c.shape}")
    return set_a, set_b, set_c

def load_before_scaling():
    """
        Load the not scaled datasets.
        This datasets have the full timesteps, the cleaned data and the data not scaled
    """
    set_a = pd.read_parquet(PROCESSED_DATA_DIR / "set_a" / "set_a_before_scaling.parquet")
    set_b = pd.read_parquet(PROCESSED_DATA_DIR / "set_b" / "set_b_before_scaling.parquet")
    set_c = pd.read_parquet(PROCESSED_DATA_DIR / "set_c" / "set_c_before_scaling.parquet")
    print("Shapes of the datasets:")
    print(f"Set A: {set_a.shape}", f"Set B: {set_b.shape}", f"Set C: {set_c.shape}")
    return set_a, set_b, set_c

def load_final_data():
    """
        Load the final datasets.
        This datasets have the full timesteps, the cleaned data, the data not scaled and the final data
    """
    set_a = pd.read_parquet(PROCESSED_DATA_DIR / "set_a" / "set_a_final.parquet")
    set_b = pd.read_parquet(PROCESSED_DATA_DIR / "set_b" / "set_b_final.parquet")
    set_c = pd.read_parquet(PROCESSED_DATA_DIR / "set_c" / "set_c_final.parquet")
    print("Shapes of the datasets:")
    print(f"Set A: {set_a.shape}", f"Set B: {set_b.shape}", f"Set C: {set_c.shape}")
    return set_a, set_b, set_c

def load_final_data_without_ICU():
    """
        Load the final datasets.
        This datasets have the full timesteps, the cleaned data, the data not scaled and the final data
    """
    set_a = pd.read_parquet(PROCESSED_DATA_DIR / "set_a" / "set_a_final.parquet")
    set_b = pd.read_parquet(PROCESSED_DATA_DIR / "set_b" / "set_b_final.parquet")
    set_c = pd.read_parquet(PROCESSED_DATA_DIR / "set_c" / "set_c_final.parquet")

    # Remove the ICUType column from the three datasets
    set_a = set_a.drop(columns=["ICUType"])
    set_b = set_b.drop(columns=["ICUType"])
    set_c = set_c.drop(columns=["ICUType"])
    print("Shapes of the datasets:")
    print(f"Set A: {set_a.shape}", f"Set B: {set_b.shape}", f"Set C: {set_c.shape}")
    return set_a, set_b, set_c



def load_outcomes():
    # Define file names
    file_names = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]

    # Directory path
    base_path = PROJ_ROOT / "data" / "data_1" / "predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0"

    # Read files into DataFrames containing all variables
    outcomes_a, outcomes_b, outcomes_c = [pd.read_csv(base_path / name) for name in file_names]

    # Extract only the "RecordID" and "In-hospital_death" column into separate DataFrames
    death_a, death_b, death_c = [df[["RecordID", "In-hospital_death"]] for df in [outcomes_a, outcomes_b, outcomes_c]]
    print(f"Shapes of labels:")
    print(f"Set A: {death_a.shape}", f"Set B: {death_b.shape}", f"Set C: {death_c.shape}")
    return death_a, death_b, death_c

__all__ = ["load_initial_data", "load_before_imputation", "load_before_ffill", "load_before_backward", "load_before_scaling", "load_final_data", "load_final_data_without_ICU", "load_outcomes"]
