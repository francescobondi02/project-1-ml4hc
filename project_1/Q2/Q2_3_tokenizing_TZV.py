################################################################################
# * Q2.3b: Tokenizing Time-series Data and Transformers
# * This script tokenizes the time-series data using TZV format and trains a Transformer model.
################################################################################

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import subprocess
from torch.utils.data import DataLoader, TensorDataset

from project_1.config import PROJ_ROOT, PROCESSED_DATA_DIR
from project_1.loading import *
from project_1.dataset import *
from project_1.features import *

torch.manual_seed(0)

################################################################################
# ^ nvidia-smi
################################################################################


def check_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, text=True, check=True
        )
        print(result.stdout)
    except Exception as e:
        print("Error checking GPU status:", e)


################################################################################
# ^ Data Loading
################################################################################

# For this part, we need to load the initial data
"""set_a, set_b, set_c = load_before_scaling()
death_a, death_b, death_c = load_outcomes()

# Scale using basic scaling
set_a_scaled, set_b_scaled, set_c_scaled = scale_features_basic(set_a, set_b, set_c)

# Remove the ICUType feature
set_a_scaled = set_a_scaled.drop(columns=['ICUType'])
set_b_scaled = set_b_scaled.drop(columns=['ICUType'])
set_c_scaled = set_c_scaled.drop(columns=['ICUType'])"""

# For this part we need the scaled data with simple
set_a_initial, set_b_initial, set_c_initial = load_initial_data()
death_a, death_b, death_c = load_outcomes()
print(set_a_initial.shape, set_b_initial.shape, set_c_initial.shape)
print(set_a_initial.head())

################################################################################
# ^ Transformer Definition
################################################################################


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes=1,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3,
    ):
        super().__init__()
        self.input_size = input_size

        # Project input features to model dimension
        self.embedding = nn.Linear(input_size, dim_feedforward)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final classifier
        self.fc = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)

        x = self.embedding(x)  # (batch, seq_len, d_model)
        # print("After embedding:", x.shape)  # Debug print
        x = self.pos_encoder(x)
        # print("After pos encoding:", x.shape)  # Debug print
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # print("After transformer encoder:", x.shape)

        x = x.mean(dim=1)  # mean pooling over time
        # print("After pooling:", x.shape)     # Debug print
        out = self.fc(x).squeeze()  # (batch,)
        # print("After fc:", out.shape)        # Debug print
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


################################################################################
# ^ Convert Data to TZV format
################################################################################

from sklearn.preprocessing import MinMaxScaler


def build_TZV_dataframe(
    original_df, label_df, base_time="2025-03-10 00:00:00", duration_hours=48
):
    """
    Build a long-format dataframe with columns [T, Z, V, y] from an original wide dataframe.

    Parameters:
        original_df (pd.DataFrame): DataFrame with columns [RecordID, Time, f1, f2, ..., f41].
        label_df (pd.DataFrame): DataFrame with columns [RecordID, y] containing the label for each RecordID.
        base_time (str): Base time used for normalizing the Time column.
        duration_hours (int): The duration (in hours) from base_time over which Time is normalized (here, 48 hours).

    Returns:
        long_df (pd.DataFrame): Long-format dataframe with columns:
                                T: normalized time [0, 1],
                                Z: index of the feature,
                                V: scaled measurement value,
                                y: label corresponding to RecordID.
        feature_to_index (dict): Mapping from original feature names to integer indices.
    """
    # Merge the labels with the original dataframe using RecordID.
    df = original_df.copy().merge(label_df, on="RecordID", how="left")

    # Convert Time to datetime and compute normalized time T.
    df["Time"] = pd.to_datetime(df["Time"])
    start_time = pd.to_datetime(base_time)
    end_time = start_time + pd.Timedelta(hours=duration_hours)
    total_seconds = (end_time - start_time).total_seconds()
    df["T"] = (df["Time"] - start_time).dt.total_seconds() / total_seconds

    # Identify feature columns: all columns except RecordID, Time, T, and y.
    feature_cols = [
        col
        for col in df.columns
        if col not in ["RecordID", "Time", "T", "In-hospital_death"]
    ]

    # Scale each feature individually using MinMaxScaler.
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Melt the dataframe from wide to long format.
    # The id_vars ("T" and "y") are preserved for each measurement.
    long_df = pd.melt(
        df,
        id_vars=["T", "In-hospital_death"],
        value_vars=feature_cols,
        var_name="Z",
        value_name="V",
    )

    # Map feature names to indices for the "Z" column.
    feature_to_index = {feat: idx for idx, feat in enumerate(feature_cols)}
    long_df["Z"] = long_df["Z"].map(feature_to_index)

    # Sort the final dataframe by normalized time T and reset the index.
    long_df = long_df.sort_values("T").reset_index(drop=True)
    long_df = long_df.dropna(subset=["V"])

    return long_df, feature_to_index


# Build the TZV dataframes
TZV_a, feature_to_index_a = build_TZV_dataframe(set_a_initial, death_a)
TZV_b, feature_to_index_b = build_TZV_dataframe(set_b_initial, death_b)
TZV_c, feature_to_index_c = build_TZV_dataframe(set_c_initial, death_c)

print(TZV_a.shape)

################################################################################
# ^ Create DataSet and DataLoader
################################################################################

# Remove the In-hospital_death column from the TZV dataframes, but save it
y_a = TZV_a.pop("In-hospital_death")
y_b = TZV_b.pop("In-hospital_death")
y_c = TZV_c.pop("In-hospital_death")

# Convert the TZV dataframes to PyTorch tensors
X_a = torch.tensor(TZV_a[["T", "Z", "V"]].values, dtype=torch.float32)
X_b = torch.tensor(TZV_b[["T", "Z", "V"]].values, dtype=torch.float32)
X_c = torch.tensor(TZV_c[["T", "Z", "V"]].values, dtype=torch.float32)
print(X_a.shape, X_b.shape, X_c.shape)

# Create the datasets and dataloaders
from torch.utils.data import TensorDataset

dataset_a = TensorDataset(X_a, torch.tensor(y_a.values, dtype=torch.float32))
dataset_b = TensorDataset(X_b, torch.tensor(y_b.values, dtype=torch.float32))
dataset_c = TensorDataset(X_c, torch.tensor(y_c.values, dtype=torch.float32))

loader_a = DataLoader(
    dataset_a, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
loader_b = DataLoader(
    dataset_b, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)
loader_c = DataLoader(
    dataset_c, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)

################################################################################
# ^ Train the Model
################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Using GPU {current_device}: {device_name}")
else:
    print("CUDA is not available.")
print("Using device:", device)
model_tvz = TransformerClassifier(input_size=3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_tvz.parameters(), lr=0.001)

print("GPU status before training:")
check_nvidia_smi()
model_tvz = train_model_with_validation(
    model_tvz, loader_a, loader_b, criterion, optimizer, device
)

################################################################################
# ^ Evaluate the Model
################################################################################

# Evaluate the model
avg_loss, aucroc, auprc = evaluate_model(model_tvz, loader_c, criterion, device)
print(f"Test Loss: {avg_loss:.4f}, AUC-ROC: {aucroc:.4f}, AUC-PRC: {auprc:.4f}")
