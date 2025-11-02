"""
app.py
-------
Main Streamlit dashboard for the Visual Debugger for Dataset Bias and Quality.
Provides an interactive interface for:
- Dataset upload
- Optional semi-supervised labeling
- Data diagnostics (duplicates, imbalance, outliers)
- Bias/conflict analysis via self-influence
- Visualization (UMAP, influence ranking, fairness metrics)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Internal imports
from data_utils.loader import load_dataset_metadata
from data_utils.labeling import suggest_labels
from embedding.extract import extract_embeddings
from diagnostics.duplicates import find_duplicates
from diagnostics.imbalance import analyze_class_balance
from diagnostics.outliers import detect_outliers
from influence.influence import (
    compute_influence_scores,
    identify_bias_conflicting_samples,
    evaluate_fairness_metrics
)
from dashboard.viz import plot_umap, show_influence_gallery


# ---------------------------------------------------------------
# Streamlit Layout and Sidebar Controls
# ---------------------------------------------------------------

st.set_page_config(page_title="Visual Debugger: Dataset Bias & Quality", layout="wide")

st.title("🧠 Visual Debugger for Dataset Bias & Quality")
st.markdown("""
This interactive tool helps you **detect**, **visualize**, and **mitigate** dataset bias and quality issues.
Select a workflow below to begin:
""")

mode = st.sidebar.radio(
    "Choose Mode:",
    ["🧩 Label Completion", "🔍 Dataset Analysis"],
    help="Select whether to complete unlabeled data or analyze dataset bias/quality."
)

# ---------------------------------------------------------------
# LABEL COMPLETION WORKFLOW
# ---------------------------------------------------------------

if mode == "🧩 Label Completion":
    st.header("Semi-Supervised Label Completion")
    data_dir = st.text_input("📁 Enter dataset folder path (organized by class):")

    if st.button("Suggest Labels"):
        if not data_dir or not Path(data_dir).exists():
            st.error("Please provide a valid dataset path.")
        else:
            st.info("Extracting embeddings and suggesting probable labels...")
            embeddings, metadata = extract_embeddings(data_dir)
            suggested = suggest_labels(embeddings, metadata)
            st.success("Label suggestions complete ✅")

            st.dataframe(suggested.head(10))
            st.download_button(
                "Download Suggested Labels",
                suggested.to_csv(index=False).encode("utf-8"),
                "suggested_labels.csv",
                "text/csv"
            )

# ---------------------------------------------------------------
# DATASET ANALYSIS WORKFLOW
# ---------------------------------------------------------------

elif mode == "🔍 Dataset Analysis":
    st.header("Dataset Bias and Quality Analysis")
    data_dir = st.text_input("📁 Enter dataset folder path (organized by class):")

    if st.button("Run Diagnostics"):
        if not data_dir or not Path(data_dir).exists():
            st.error("Please provide a valid dataset path.")
        else:
            st.info("🔄 Loading and analyzing dataset...")

            # Load dataset & extract features
            metadata = load_dataset_metadata(data_dir)
            embeddings, _ = extract_embeddings(data_dir)

            # --- Step 1: Diagnostics ---
            duplicates = find_duplicates(embeddings, metadata)
            imbalance = analyze_class_balance(metadata)
            outliers = detect_outliers(embeddings, metadata)

            st.subheader("1️⃣ Data Diagnostics Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Duplicates", len(duplicates))
            with col2:
                st.metric("Outliers", len(outliers))
            with col3:
                st.metric("Classes", len(imbalance))

            st.dataframe(imbalance)

            # --- Step 2: Visual Embedding Space ---
            st.subheader("2️⃣ Embedding Explorer (UMAP Projection)")
            plot_umap(embeddings, metadata["label"], outliers)

            # --- Step 3: Influence and Bias Analysis ---
            st.subheader("3️⃣ Bias-Conflicting & Influential Samples")
            labels = metadata["label"].values
            sensitive_attr = metadata.get("sensitive_attr", np.random.choice(["A", "B"], len(labels)))
            influence_scores = compute_influence_scores(embeddings, labels)
            bias_samples = identify_bias_conflicting_samples(embeddings, labels, sensitive_attr)

            st.write(f"Detected {len(bias_samples)} bias-conflicting samples.")
            show_influence_gallery(metadata, bias_samples)

            # --- Step 4: Fairness Metrics ---
            st.subheader("4️⃣ Fairness Metrics (Before Correction)")
            fairness = evaluate_fairness_metrics(labels, labels, sensitive_attr)
            st.json(fairness)

            # --- Step 5: Suggestions ---
            st.subheader("5️⃣ Actionable Suggestions")
            st.markdown("""
            ✅ **Suggested actions**:
            - Review and relabel top bias-conflicting samples  
            - Remove near-duplicate or outlier samples  
            - Balance dataset using oversampling or reweighting  
            - Use fairness regularization (Equalized Odds / Demographic Parity)
            """)

            st.success("Diagnostics Complete ✅")
