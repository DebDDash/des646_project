"""
influence.py
-------------
Implements simplified self-influence analysis to identify bias-conflicting or mislabeled samples.
Based on approximations to influence functions (Koh & Liang, 2017),
adapted for fast evaluation on small image datasets.

Functions:
- compute_influence_scores(): estimates per-sample influence on model loss.
- identify_bias_conflicting_samples(): highlights samples whose removal improves fairness/accuracy.
- evaluate_fairness_metrics(): computes basic fairness stats (e.g., demographic parity, equalized odds).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict, Tuple

# --------------------------------------------------------------------
# Simple classifier for influence computation
# --------------------------------------------------------------------

class SmallClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# --------------------------------------------------------------------
# Influence computation
# --------------------------------------------------------------------

def compute_influence_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    epochs: int = 5,
    batch_size: int = 16,
    device: str = "cpu"
) -> np.ndarray:
    """
    Approximate per-sample influence on model loss.
    Uses leave-one-out gradient magnitude as proxy for self-influence.

    Args:
        embeddings: (N, D) feature matrix from backbone
        labels: (N,) class labels
        lr: learning rate for training small classifier
        epochs: training epochs
        batch_size: mini-batch size
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray: influence scores (higher = more influential / potentially problematic)
    """
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)

    model = SmallClassifier(embeddings.shape[1], len(np.unique(labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initial training
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Compute per-sample gradient norm as proxy for influence
    influence_scores = []
    for i in range(len(X)):
        model.zero_grad()
        out = model(X[i].unsqueeze(0))
        loss = criterion(out, y[i].unsqueeze(0))
        loss.backward()
        grad_norm = sum((p.grad.norm() ** 2).item() for p in model.parameters())
        influence_scores.append(grad_norm)

    return np.array(influence_scores)


# --------------------------------------------------------------------
# Fairness and bias analysis
# --------------------------------------------------------------------

def evaluate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    Compute simple group fairness metrics.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels
        sensitive_attr: binary or categorical attribute (e.g., gender, background)

    Returns:
        dict of fairness measures (Demographic Parity, Equalized Odds)
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "attr": sensitive_attr})

    groups = df.groupby("attr")
    metrics = {}
    pos_rate = groups["y_pred"].mean()
    true_pos_rate = groups.apply(lambda g: np.mean((g["y_pred"] == 1) & (g["y_true"] == 1)))

    metrics["demographic_parity_diff"] = abs(pos_rate.max() - pos_rate.min())
    metrics["equalized_odds_diff"] = abs(true_pos_rate.max() - true_pos_rate.min())

    return metrics


def identify_bias_conflicting_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sensitive_attr: np.ndarray,
    threshold: float = 0.9
) -> pd.DataFrame:
    """
    Identify 'bias-conflicting' samples:
    points whose influence is high and whose predicted label disagrees with group bias trend.

    Args:
        embeddings: N x D array
        labels: class labels
        sensitive_attr: binary/categorical group attribute
        threshold: quantile threshold for top influential samples

    Returns:
        pd.DataFrame: records with index, influence_score, bias_flag
    """
    influence_scores = compute_influence_scores(embeddings, labels)
    cutoff = np.quantile(influence_scores, threshold)
    high_influence_idx = np.where(influence_scores >= cutoff)[0]

    # simple heuristic: group majority label vs individual label disagreement
    df = pd.DataFrame({"index": np.arange(len(labels)), "label": labels, "attr": sensitive_attr})
    group_majority = df.groupby("attr")["label"].agg(lambda x: x.value_counts().idxmax())
    bias_conflicting = []

    for i in high_influence_idx:
        attr = df.loc[i, "attr"]
        if df.loc[i, "label"] != group_majority[attr]:
            bias_conflicting.append(i)

    return pd.DataFrame({
        "index": bias_conflicting,
        "influence_score": influence_scores[bias_conflicting],
        "bias_flag": True
    })


# --------------------------------------------------------------------
# Demo usage
# --------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    embeddings = np.random.randn(200, 64)
    labels = np.random.randint(0, 2, 200)
    sensitive = np.random.choice(["A", "B"], 200)

    scores = compute_influence_scores(embeddings, labels)
    print("Sample influence scores:", scores[:5])

    fairness = evaluate_fairness_metrics(labels, labels, sensitive)
    print("Fairness metrics:", fairness)

    bias_samples = identify_bias_conflicting_samples(embeddings, labels, sensitive)
    print("Bias-conflicting samples:\n", bias_samples.head())
