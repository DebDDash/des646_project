"""
This module provides quantitative measures of dataset diversity and inter-class overlap
using embedding-based statistical metrics. It helps determine whether certain classes
are overly homogeneous (low diversity) or overlapping with others (potential bias/confusion).

Functions:
- compute_intra_class_diversity(): measures per-class embedding variance.
- compute_inter_class_overlap(): measures overlap using Fréchet distance and JSD.
- compute_diversity_index(): combines intra/inter metrics into a summary report.
- plot_diversity_heatmap(): optional visualization for inter-class overlap.

Dependencies:
    numpy, pandas, scipy, sklearn, matplotlib (optional for visualization)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


# -----------------------------
# 1️⃣ Intra-Class Diversity
# -----------------------------
def compute_intra_class_diversity(embeddings, labels):
    """
    Compute per-class diversity using mean pairwise embedding distance
    and variance (higher = more diverse class).
    
    Args:
        embeddings (np.ndarray): shape (N, D) - feature embeddings.
        labels (list or np.ndarray): class labels for each embedding.
    Returns:
        pd.DataFrame: diversity metrics per class.
    """
    classes = np.unique(labels)
    diversity_stats = []

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        cls_emb = embeddings[cls_idx]
        if len(cls_emb) < 2:
            continue
        
        # Compute pairwise distances
        dist_matrix = pairwise_distances(cls_emb)
        mean_div = np.mean(dist_matrix)
        var_div = np.var(dist_matrix)
        
        diversity_stats.append({
            "class": cls,
            "n_samples": len(cls_emb),
            "mean_distance": mean_div,
            "variance_distance": var_div
        })

    return pd.DataFrame(diversity_stats)


# -----------------------------
# 2️⃣ Inter-Class Overlap
# -----------------------------
def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean = np.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_inter_class_overlap(embeddings, labels):
    """
    Estimate overlap between class distributions using Fréchet distance
    (small = overlapping, large = well-separated).
    
    Args:
        embeddings (np.ndarray): (N, D)
        labels (np.ndarray): (N,)
    Returns:
        pd.DataFrame: symmetric matrix (classes × classes) of overlap scores.
    """
    classes = np.unique(labels)
    n = len(classes)
    overlap_matrix = np.zeros((n, n))

    class_stats = {}
    for cls in classes:
        cls_emb = embeddings[labels == cls]
        class_stats[cls] = {
            "mu": np.mean(cls_emb, axis=0),
            "sigma": np.cov(cls_emb, rowvar=False)
        }

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i >= j:
                continue
            f_dist = frechet_distance(class_stats[c1]["mu"], class_stats[c1]["sigma"],
                                      class_stats[c2]["mu"], class_stats[c2]["sigma"])
            overlap_matrix[i, j] = overlap_matrix[j, i] = f_dist

    return pd.DataFrame(overlap_matrix, index=classes, columns=classes)


# -----------------------------
# 3️⃣ Combined Diversity Index
# -----------------------------
def compute_diversity_index(embeddings, labels):
    """
    Aggregate diversity metrics into a summary score.
    Higher intra-class variance and lower inter-class overlap => high diversity index.
    
    Returns:
        dict: summary metrics.
    """
    intra_df = compute_intra_class_diversity(embeddings, labels)
    inter_df = compute_inter_class_overlap(embeddings, labels)

    mean_intra = intra_df["mean_distance"].mean()
    mean_inter = inter_df.values[np.triu_indices_from(inter_df, k=1)].mean()

    diversity_index = mean_intra / (mean_inter + 1e-8)
    
    return {
        "mean_intra_distance": mean_intra,
        "mean_inter_distance": mean_inter,
        "diversity_index": diversity_index
    }


# -----------------------------
# 4️⃣ Visualization
# -----------------------------
def plot_diversity_heatmap(overlap_df, cmap="coolwarm"):
    """Visualize inter-class overlap as heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(overlap_df, cmap=cmap)
    plt.xticks(range(len(overlap_df.columns)), overlap_df.columns, rotation=45)
    plt.yticks(range(len(overlap_df.index)), overlap_df.index)
    plt.title("Inter-Class Overlap (Fréchet Distance)")
    plt.colorbar(label="Distance")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Simulated example
    np.random.seed(42)
    embeddings = np.vstack([
        np.random.normal(0, 1, (50, 10)),
        np.random.normal(3, 1, (50, 10)),
        np.random.normal(-2, 0.5, (50, 10))
    ])
    labels = np.array(["A"]*50 + ["B"]*50 + ["C"]*50)

    intra = compute_intra_class_diversity(embeddings, labels)
    inter = compute_inter_class_overlap(embeddings, labels)
    summary = compute_diversity_index(embeddings, labels)

    print("Intra-Class Diversity:\n", intra)
    print("\nInter-Class Overlap:\n", inter)
    print("\nSummary Diversity Index:", summary)

    plot_diversity_heatmap(inter)
