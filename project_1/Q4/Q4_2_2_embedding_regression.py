import numpy as np
import pandas as pd
from ollama import embed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Function to load embeddings
def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['In-hospital_death']).values
    y = df['In-hospital_death'].values
    return X, y


def train_and_evaluate(train_X, train_y, test_X, test_y):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train_X, train_y)
    probs = clf.predict_proba(test_X)[:, 1]

    auroc = roc_auc_score(test_y, probs)
    auprc = average_precision_score(test_y, probs)
    return probs, auroc, auprc

def visualize_tsne(embeddings, labels, title="t-SNE of Embeddings"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label="In-hospital Death (0/1)")
    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.savefig("llm_embeddings_tse_plot.png", bbox_inches='tight', dpi=300)  # Save as high-res PNG
    plt.show()

# Load embeddings
train_X, train_y = load_embeddings("embeddings_a.csv")
test_X, test_y = load_embeddings("embeddings_c.csv")

probs, auroc, auprc = train_and_evaluate(train_X, train_y, test_X, test_y)
print(f"Test AuROC: {auroc:.4f}, Test AuPRC: {auprc:.4f}")

visualize_tsne(test_X, test_y, title=f"t-SNE Embeddings (Test Set)")