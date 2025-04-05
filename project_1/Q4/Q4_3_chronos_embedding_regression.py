#########################################################################################################################
# * Q4.3: Embedding Regression
# * This script loads the embeddings from CSV files, converts them using Chronos,
# * trains a logistic regression model on the training set and evaluates it on the test set.
#########################################################################################################################

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd
import torch
from chronos import ChronosPipeline

from project_1.config import PROCESSED_DATA_DIR
from project_1.loading import load_before_scaling, load_outcomes


#################################################################
# ^ Naive Embedding Generation
#################################################################


def get_univariate_embedding(pipeline, single_var_series):
    """
    Given a single univariate time series (1D array), return its Chronos embedding.
    """
    if not isinstance(single_var_series, torch.Tensor):
        series_tensor = torch.tensor(single_var_series, dtype=torch.float32)
    else:
        series_tensor = single_var_series.float()
    # Chronos supports single 1D tensors directly
    with torch.no_grad():
        embedding, _ = pipeline.embed(series_tensor)

    if (
        embedding.ndim == 3
    ):  # bc of batch logic it returns embedding as torch.Size([1, 49, 512])
        embedding = embedding.squeeze(0)
    return embedding.mean(dim=0)  # Average pooling


def compute_patient_embedding_naive(pipeline, multivar_data, variables):
    """
    Given a dictionary of { variable_name: 1D time series } for a patient,
    compute the average embedding across variables.
    """
    embeddings = []

    for var_name, series in multivar_data.items():
        if var_name in variables:
            if len(series) == 0:
                continue
            emb = get_univariate_embedding(pipeline, series)
            embeddings.append(emb)

    if not embeddings:
        return torch.zeros(512)

    return torch.stack(embeddings, dim=0).mean(dim=0)  # shape: [embedding_dim]


def build_patient_embeddings_naive(pipeline, df_timeseries, df_labels, variables):
    """
    Builds Chronos embeddings per patient using naive variable-wise averaging.

    Args:
        pipeline: ChronosPipeline instance
        df_timeseries: long-form time series DataFrame with columns:
                       ['RecordID', 'Time', var1, var2, ...]
        df_labels: DataFrame with columns ['RecordID', 'In-hospital_death']
        variables: list of variable names to embed (e.g., ['HeartRate', 'WBC'])

    Returns:
        X: numpy array of shape [num_patients, embedding_dim]
        y: numpy array of shape [num_patients]
    """
    X_list = []
    y_list = []

    for rid, group in df_timeseries.groupby("RecordID"):
        patient_data = {}

        for var in variables:
            if var in group.columns:
                series = group[var].dropna().values
                if len(series) > 0:
                    patient_data[var] = series

        if not patient_data:
            continue  # skip if no valid data

        emb = compute_patient_embedding_naive(pipeline, patient_data, variables)
        X_list.append(emb.cpu().numpy())

        # Lookup label
        label_row = df_labels[df_labels["RecordID"] == rid]
        if not label_row.empty:
            y_list.append(label_row["In-hospital_death"].values[0])
        else:
            continue  # skip if label missing

    return np.vstack(X_list), np.array(y_list)


##################################################################
# ^ Simple Neural Aggregator Approach
##################################################################


class ChannelAggregator(nn.Module):
    def __init__(self, num_vars, embed_dim=512, hidden_dim=64, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * num_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def build_multivar_embeddings(pipeline, patient_df, variables):
    """
    Takes a single patient’s time series (dataframe) and computes a stacked tensor of embeddings.
    Returns: torch.Tensor of shape [num_vars, embed_dim]
    """
    embeddings = []
    for var in variables:
        if var in patient_df.columns:
            series = patient_df[var].dropna().values
            if len(series) == 0:
                continue
            with torch.no_grad():
                emb, _ = pipeline.embed(torch.tensor(series, dtype=torch.float32))
                if emb.ndim == 3:
                    emb = emb.squeeze(0)
                emb = emb.mean(dim=0)  # [D]
                embeddings.append(emb.unsqueeze(0))  # [1, D]

    if not embeddings:
        return torch.zeros((1, 512))  # fallback for no valid data
    return torch.cat(embeddings, dim=0)  # [num_vars, D]


def train_aggregator(
    pipeline,
    aggregator,
    df_timeseries,
    df_labels,
    variables,
    epochs=5,
    lr=1e-3,
    device="cpu",
):
    aggregator.to(device)
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_list, y_list = [], []
    for rid, group in df_timeseries.groupby("RecordID"):
        emb_stack = build_multivar_embeddings(pipeline, group, variables)
        if emb_stack.shape[0] < 1:
            continue  # skip patients with no valid data

        label_row = df_labels[df_labels["RecordID"] == rid]
        if label_row.empty:
            continue
        label = label_row["In-hospital_death"].values[0]

        X_list.append(emb_stack.unsqueeze(0))  # [1, num_vars, D]
        y_list.append(label)

    X_tensor = torch.cat(X_list, dim=0).float().to(device)  # [batch, num_vars, D]
    y_tensor = (
        torch.tensor(y_list, dtype=torch.float32).unsqueeze(-1).to(device)
    )  # [batch, 1]

    for epoch in range(epochs):
        aggregator.train()
        optimizer.zero_grad()
        logits = aggregator(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss={loss.item():.4f}")

    return aggregator


def evaluate_aggregator(
    pipeline, aggregator, df_timeseries, df_labels, variables, device="cpu"
):
    aggregator.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for rid, group in df_timeseries.groupby("RecordID"):
            emb_stack = build_multivar_embeddings(pipeline, group, variables)
            if emb_stack.shape[0] < 1:
                continue

            label_row = df_labels[df_labels["RecordID"] == rid]
            if label_row.empty:
                continue
            label = label_row["In-hospital_death"].values[0]

            logit = aggregator(emb_stack.unsqueeze(0).to(device))  # [1, 1]
            all_logits.append(logit.item())
            all_labels.append(label)

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    probs = 1 / (1 + np.exp(-all_logits))

    auroc = roc_auc_score(all_labels, probs)
    auprc = average_precision_score(all_labels, probs)
    return auroc, auprc


##################################################################
# ^ Main Function
##################################################################

if __name__ == "__main__":

    # ----------------------------- IMPORT AND RUNNING PARAMETERS ------------------

    run_efficient = False

    df_a, df_b, df_c = load_before_scaling()
    death_a, death_b, death_c = load_outcomes()

    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    list_of_variables = [
        "Gender",
        "Height",
        "Weight",
        "Age",
        "Albumin",
        "Cholesterol",
        "DiasABP",
        "HCO3",
        "HCT",
        "HR",
        "Mg",
        "MAP",
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "SysABP",
        "PaCO2",
        "PaO2",
        "Platelets",
        "RespRate",
        "Temp",
        "ALP",
        "ALT",
        "AST",
        "BUN",
        "Bilirubin",
        "Creatinine",
        "FiO2",
        "GCS",
        "Glucose",
        "K",
        "Lactate",
        "MechVent",
        "Urine",
        "WBC",
        "pH",
        "SaO2",
        "TroponinT",
        "TroponinI",
    ]

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device_map,
        torch_dtype=torch.float32,
    )

    df_a_small = df_a.sample(100, random_state=1)
    df_c_small = df_c.sample(100, random_state=1)

    # ----------------------------- 1) NAIVE APPROACH ---------------------------------

    if run_efficient:
        train_X, train_y = build_patient_embeddings_naive(
            pipeline, df_a_small, death_a, list_of_variables
        )
        test_X, test_y = build_patient_embeddings_naive(
            pipeline, df_c_small, death_c, list_of_variables
        )
    else:
        train_X, train_y = build_patient_embeddings_naive(
            pipeline, df_a, death_a, list_of_variables
        )
        test_X, test_y = build_patient_embeddings_naive(
            pipeline, df_c, death_c, list_of_variables
        )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(train_X, train_y)
    probs = clf.predict_proba(test_X)[:, 1]
    auroc = roc_auc_score(test_y, probs)
    auprc = average_precision_score(test_y, probs)
    print(f"[Naive Aggregation] AuROC={auroc:.4f}, AuPRC={auprc:.4f}")

    # ------------------------ 2) MLP APPROACH ----------------------------------------

    aggregator = ChannelAggregator(
        num_vars=len(list_of_variables), embed_dim=512, hidden_dim=64, out_dim=1
    )

    # Train
    aggregator = train_aggregator(
        pipeline=pipeline,
        aggregator=aggregator,
        df_timeseries=df_a_small if run_efficient else df_a,
        df_labels=death_a,
        variables=list_of_variables,
        epochs=5,
        lr=1e-3,
        device=device_map,
    )

    # Evaluate
    auroc, auprc = evaluate_aggregator(
        pipeline=pipeline,
        aggregator=aggregator,
        df_timeseries=df_c_small if run_efficient else df_c,
        df_labels=death_c,
        variables=list_of_variables,
        device=device_map,
    )

    print(f"[Neural Aggregator] AuROC={auroc:.4f}, AuPRC={auprc:.4f}")
