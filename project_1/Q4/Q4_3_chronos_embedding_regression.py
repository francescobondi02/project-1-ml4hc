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


def build_multivar_embeddings(pipeline, patient_df, variables):
    """
    For a given patient (dataframe), compute the embeddings for each variable as a
    time series, without averaging. For each variable in 'variables', this returns a tensor
    of shape [L, embedding_dim] (where L is the number of time points) or None if no data.
    """
    embeddings_list = []
    for var in variables:
        if var in patient_df.columns:
            series = patient_df[var].dropna().values
            if len(series) == 0:
                embeddings_list.append(None)
            else:
                with torch.no_grad():
                    emb, _ = pipeline.embed(torch.tensor(series, dtype=torch.float32))
                    # Expected shape [1, L, embedding_dim]; squeeze batch if needed.
                    if emb.ndim == 3:
                        emb = emb.squeeze(0)  # now [L, embedding_dim]
                    embeddings_list.append(emb)
        else:
            embeddings_list.append(None)
    return embeddings_list


class TemporalChannelAggregator(nn.Module):
    """
    This aggregator processes each variable's time series with a GRU to capture
    temporal dynamics. The GRU outputs a summary vector (using its final hidden state)
    per variable. All variable summaries are concatenated and passed through an MLP
    to output the logit for the prediction.
    """

    def __init__(self, num_vars, embed_dim=512, gru_hidden_dim=64, final_hidden_dim=64, out_dim=1):
        super().__init__()
        self.num_vars = num_vars
        self.gru_hidden_dim = gru_hidden_dim
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(num_vars * gru_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Linear(final_hidden_dim, out_dim)
        )

    def forward(self, embeddings_list):
        summaries = []
        for emb in embeddings_list:
            if emb is None or emb.size(0) == 0:
                # If missing, use a zero vector summary.
                # (Assume device same as first non-None tensor; otherwise, defaults to CPU.)
                device = embeddings_list[0].device if (embeddings_list[0] is not None) else torch.device("cpu")
                summary = torch.zeros(self.gru_hidden_dim, device=device)
            else:
                # emb: [L, embed_dim] --> add batch dimension: [1, L, embed_dim]
                emb_seq = emb.unsqueeze(0)
                _, h_n = self.gru(emb_seq)  # h_n: [1, 1, gru_hidden_dim]
                summary = h_n.squeeze(0).squeeze(0)  # shape [gru_hidden_dim]
            summaries.append(summary)
        # Concatenate all variable summaries: [num_vars * gru_hidden_dim]
        concatenated = torch.cat(summaries, dim=-1)
        logit = self.mlp(concatenated)  # [out_dim]
        return logit


def train_temporal_aggregator_from_raw(pipeline, aggregator, df_timeseries, df_labels, variables, epochs=5, lr=1e-3,
                                       device='cpu'):
    """
    Builds training samples per patient by computing time series embeddings for each variable.
    Then trains the aggregator using an individual sample per patient.
    """
    aggregator.to(device)
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Precompute embeddings for each patient
    samples = []
    grouped = df_timeseries.groupby("RecordID")
    for rid, group in tqdm(grouped, desc="Building training embeddings"):
        embeddings_list = build_multivar_embeddings(pipeline, group, variables)
        label_row = df_labels[df_labels["RecordID"] == rid]
        if label_row.empty:
            continue
        label = label_row["In-hospital_death"].values[0]
        samples.append((embeddings_list, label))

    for epoch in range(epochs):
        epoch_loss = 0.0
        aggregator.train()
        for embeddings_list, label in samples:
            # Ensure each embedding tensor is on the correct device.
            embeddings_list = [emb.to(device) if emb is not None else None for emb in embeddings_list]
            optimizer.zero_grad()
            logit = aggregator(embeddings_list)
            target = torch.tensor([label], dtype=torch.float32, device=device)
            loss = criterion(logit.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss={epoch_loss / len(samples):.4f}")

    return aggregator


def evaluate_temporal_aggregator_from_raw(pipeline, aggregator, df_timeseries, df_labels, variables, device='cpu'):
    """
    Evaluates the trained aggregator on the validation/test set.
    """
    aggregator.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for rid, group in df_timeseries.groupby("RecordID"):
            embeddings_list = build_multivar_embeddings(pipeline, group, variables)
            label_row = df_labels[df_labels["RecordID"] == rid]
            if label_row.empty:
                continue
            label = label_row["In-hospital_death"].values[0]
            embeddings_list = [emb.to(device) if emb is not None else None for emb in embeddings_list]
            logit = aggregator(embeddings_list)
            all_logits.append(logit.item())
            all_labels.append(label)
    probs = 1 / (1 + np.exp(-np.array(all_logits)))
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

    aggregator = TemporalChannelAggregator(
        num_vars=len(list_of_variables),
        embed_dim=embedding_dim,
        gru_hidden_dim=64,
        final_hidden_dim=64,
        out_dim=1
    )

    # Train
    aggregator = train_temporal_aggregator_from_raw(
        pipeline=pipeline,
        aggregator=aggregator,
        df_timeseries=df_a_small if run_efficient else df_a,
        df_labels=death_a,
        variables=list_of_variables,
        epochs=5,
        lr=1e-3,
        device=device_map
    )

    # Evaluate
    auroc, auprc = evaluate_temporal_aggregator_from_raw(
        pipeline=pipeline,
        aggregator=aggregator,
        df_timeseries=df_c_small if run_efficient else df_c,
        df_labels=death_c,
        variables=list_of_variables,
        device=device_map
    )

    print(f"[Neural Aggregator] AuROC={auroc:.4f}, AuPRC={auprc:.4f}")
