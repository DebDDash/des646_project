import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ---------------------------------------------------------------------
# Small Classifier
# ---------------------------------------------------------------------
class SmallClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------
# Influence Computation
# ---------------------------------------------------------------------
def compute_influence_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    epochs: int = 5,
    batch_size: int = 16,
    device: str = "cpu"
) -> np.ndarray:
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)

    model = SmallClassifier(embeddings.shape[1], len(np.unique(labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    influence_scores = []
    for i in range(len(X)):
        model.zero_grad()
        out = model(X[i].unsqueeze(0))
        loss = criterion(out, y[i].unsqueeze(0))
        loss.backward()
        grad_norm = sum((p.grad.norm() ** 2).item() for p in model.parameters())
        influence_scores.append(grad_norm)

    return np.array(influence_scores)


# ---------------------------------------------------------------------
# Fairness and Bias Analysis
# ---------------------------------------------------------------------
def evaluate_fairness_metrics(y_true, y_pred, sensitive_attr):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "attr": sensitive_attr})
    groups = df.groupby("attr")

    metrics = {}
    pos_rate = groups["y_pred"].mean()
    true_pos_rate = groups.apply(lambda g: np.mean((g["y_pred"] == 1) & (g["y_true"] == 1)))

    metrics["demographic_parity_diff"] = abs(pos_rate.max() - pos_rate.min())
    metrics["equalized_odds_diff"] = abs(true_pos_rate.max() - true_pos_rate.min())

    return metrics


# ---------------------------------------------------------------------
# Identify Bias-Conflicting Samples
# ---------------------------------------------------------------------
def identify_bias_conflicting_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sensitive_attr: np.ndarray,
    threshold: float = 0.9
) -> pd.DataFrame:
    influence_scores = compute_influence_scores(embeddings, labels)
    cutoff = np.quantile(influence_scores, threshold)
    high_influence_idx = np.where(influence_scores >= cutoff)[0]

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


