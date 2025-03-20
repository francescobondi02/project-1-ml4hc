import pandas as pd

# Read the Parquet file
data = pd.read_parquet("data/processed/set_a.parquet")

# Display the first few rows
print(data.head())