#########################################################################################################################
# * Q4.2.1: Embedding Generation
# * This script generates embeddings for patient summaries using the Ollama library.
# * It processes the summaries from different datasets (A, B, C) and saves the embeddings along with their labels to CSV files.
# * The script uses the `gemma2:9b` model for generating embeddings.
#########################################################################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
from ollama import embed

from project_1.config import PROCESSED_DATA_DIR

##########################################################################################################################
# ^ Utility Functions
##########################################################################################################################


def get_embedding_ollama(text: str, model_name):
    """
    Calls the Ollama embedding API for a given string.
    """
    response = embed(model=model_name, input=text)
    return np.array(response.get("embeddings", [])[0], dtype=float)


def build_embeddings(df, summary_col="summary_trend", model_name="gemma2:9b"):
    """
    For each row in the DataFrame, get an embedding and corresponding label.
    Shows progress using tqdm.
    """
    embeddings = []
    labels = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Embedding [{summary_col}] with {model_name}",
    ):
        text = row[summary_col]
        label = row["In-hospital_death"]
        emb = get_embedding_ollama(text, model_name=model_name)
        embeddings.append(emb)
        labels.append(label)

    return np.vstack(embeddings), np.array(labels)


###########################################################################################################################
# ^ Main Function
###########################################################################################################################

if __name__ == "__main__":
    print("🔄 Loading data...")
    summaries_a = pd.read_csv(PROCESSED_DATA_DIR / "set_a" / "summaries_a.csv")
    summaries_b = pd.read_csv(PROCESSED_DATA_DIR / "set_b" / "summaries_b.csv")
    summaries_c = pd.read_csv(PROCESSED_DATA_DIR / "set_c" / "summaries_c.csv")

    print("🚀 Building embeddings for summaries_a...")
    embeddings_a, labels_a = build_embeddings(summaries_a)

    print("🚀 Building embeddings for summaries_c...")
    embeddings_c, labels_c = build_embeddings(summaries_c)

    # Convert to DataFrames
    df_a = pd.DataFrame(embeddings_a)
    df_a["In-hospital_death"] = labels_a

    df_c = pd.DataFrame(embeddings_c)
    df_c["In-hospital_death"] = labels_c

    print("💾 Saving to CSV...")
    df_a.to_csv(PROCESSED_DATA_DIR / "set_a" / "embeddings_a.csv", index=False)
    df_c.to_csv(PROCESSED_DATA_DIR / "set_c" / "embeddings_c.csv", index=False)

    print("✅ Done.")
