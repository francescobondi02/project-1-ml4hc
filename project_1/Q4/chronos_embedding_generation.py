import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from chronos import ChronosPipeline
from tqdm import tqdm

# ================================
# Configuration
# ================================
device_map = "cuda" if torch.cuda.is_available() else "cpu"
embedding_dim = 512

list_of_variables = ['Gender', 'Height', 'Weight', 'Age', 'Albumin',
       'Cholesterol', 'DiasABP', 'HCO3', 'HCT', 'HR', 'Mg', 'MAP', 'Na',
       'NIDiasABP', 'NIMAP', 'NISysABP', 'SysABP', 'PaCO2', 'PaO2',
       'Platelets', 'RespRate', 'Temp', 'ALP', 'ALT', 'AST', 'BUN',
       'Bilirubin', 'Creatinine', 'FiO2', 'GCS', 'Glucose', 'K', 'Lactate',
       'MechVent', 'Urine', 'WBC', 'pH', 'SaO2', 'TroponinT', 'TroponinI']

# ================================
# Load Chronos
# ================================
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=device_map,
    torch_dtype=torch.float32,
)

# ================================
# Efficient Embedding Functions
# ================================

def embed_series_batch(series_list):
    """
    Batch-embed a list of univariate time series.
    Each item: a 1D numpy array or tensor.
    Returns: list of pooled embeddings (one per input)
    """
    tensors = [torch.tensor(s, dtype=torch.float32) for s in series_list]
    with torch.no_grad():
        embeddings, _ = pipeline.embed(tensors)  # [batch, T, D]
        pooled = embeddings.mean(dim=1)          # [batch, D]
    return pooled

def build_naive_embeddings_fast(pipeline, df_ts, df_labels, variables):
    label_lookup = df_labels.set_index("RecordID")["In-hospital_death"].to_dict()
    results = []

    grouped = df_ts.groupby("RecordID")
    for rid, group in tqdm(grouped, desc="Naive Embeddings"):
        if rid not in label_lookup:
            continue

        patient_series = [group[var].dropna().values for var in variables if var in group.columns and group[var].notna().any()]
        if not patient_series:
            continue

        pooled = embed_series_batch(patient_series)  # shape: [num_vars, D]
        avg_embedding = pooled.mean(dim=0).cpu().numpy()  # shape: [D]
        label = label_lookup[rid]
        results.append(np.concatenate([avg_embedding, [label]]))

    return pd.DataFrame(results, columns=[f"f{i}" for i in range(embedding_dim)] + ["label"])

def build_multivar_embeddings_fast(pipeline, df_ts, df_labels, variables):
    label_lookup = df_labels.set_index("RecordID")["In-hospital_death"].to_dict()
    results = []

    grouped = df_ts.groupby("RecordID")
    for rid, group in tqdm(grouped, desc="Multivariate Embeddings"):
        if rid not in label_lookup:
            continue

        patient_series = [group[var].dropna().values for var in variables if var in group.columns and group[var].notna().any()]
        if not patient_series:
            continue

        pooled = embed_series_batch(patient_series)  # [num_vars, D]
        flat = pooled.flatten().cpu().numpy()  # [num_vars * D]
        label = label_lookup[rid]
        results.append(np.concatenate([flat, [label]]))

    embed_dim = pooled.shape[-1]
    num_vars = pooled.shape[0]
    return pd.DataFrame(results, columns=[f"f{i}" for i in range(num_vars * embed_dim)] + ["label"])

# ================================
# Load Your Data
# ================================
df_a = pd.read_csv("df_a.csv")      # long format: RecordID, Time, var1, var2, ...
df_c = pd.read_csv("df_c.csv")
death_a = pd.read_csv("death_a.csv")       # columns: RecordID, In-hospital_death
death_c = pd.read_csv("death_c.csv")

# ================================
# Run & Save Naive Embeddings
# ================================
print(device_map)
print("Building naive Chronos embeddings...")
naive_train = build_naive_embeddings_fast(pipeline, df_a, death_a, list_of_variables)
naive_test = build_naive_embeddings_fast(pipeline, df_c, death_c, list_of_variables)

naive_train.to_csv("chronos_naive_train.csv", index=False)
naive_test.to_csv("chronos_naive_test.csv", index=False)
print("✅ Saved naive Chronos embeddings.")

# ================================
# Run & Save Multivariate Embeddings
# ================================
print("Building multivariate Chronos embeddings...")
mlp_train = build_multivar_embeddings_fast(pipeline, df_a, death_a, list_of_variables)
mlp_test = build_multivar_embeddings_fast(pipeline, df_c, death_c, list_of_variables)

mlp_train.to_csv("chronos_mlp_train.csv", index=False)
mlp_test.to_csv("chronos_mlp_test.csv", index=False)
print("✅ Saved multivariate Chronos embeddings.")
