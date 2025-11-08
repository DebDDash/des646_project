import os
import sys
import shutil
import tempfile
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import json

# Add parent directory to Python path (so we can import data_utils)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.loader import load_dataset, bulk_extract_metadata, summarize_dataset
from data_utils import visualize as viz
from embedding import visualize as embviz
from data_utils.labelling import semi_supervised_labeling
from embedding.extract import EmbeddingExtractor
from embedding.indexer import build_faiss_index_auto, find_duplicates_faiss_fast, build_knn_graph_fast
from embedding.visualize import reduce_embeddings, plot_embedding_scatter, plot_similarity_heatmap
from diagnostics.diversity import compute_intra_class_diversity, compute_inter_class_overlap, compute_diversity_index, plot_diversity_heatmap
from diagnostics.duplicates import find_duplicates, summarize_duplicates
from diagnostics.imbalance import summarize_imbalance
from diagnostics.outliers import summarize_outliers
from diagnostics.fix_dataset import generate_repair_suggestions, apply_repairs

# Influence analysis
from influence.influence import (
    compute_influence_scores,
    evaluate_fairness_metrics,
    identify_bias_conflicting_samples,
)
from influence.sensitive_attr import create_sensitive_attribute_file, SENSITIVE_ATTR_PATH


st.set_page_config(page_title="Visual Debugger Dashboard", layout="wide")

st.title("🧠 Visual Debugger Dashboard")
st.write("Inspect, diagnose, and visualize dataset and embedding statistics.")

# ---- Sidebar ----
mode = st.sidebar.radio(
    "Choose Mode:",
    ["Upload & Label Dataset", "Run Diagnostics"],
    index=0
)

output_dir = "testing/output"

# ---- Mode 1: Upload & Label ----
if mode == "Upload & Label Dataset":
    st.header("📂 Upload and Label Dataset")

    upload_type = st.radio(
        "Select upload type:",
        ["ZIP folder (recommended)", "Individual images / CSV files"]
    )

    dataset_path = None

    # --- ZIP Upload ---
    if upload_type == "ZIP folder (recommended)":
        uploaded_zip = st.file_uploader("Upload your dataset as a ZIP file:", type=["zip"])
        if uploaded_zip:
            tmp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(tmp_dir, uploaded_zip.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            items = os.listdir(tmp_dir)
            st.success(f"✅ Extracted ZIP to temporary folder.")
            st.write("Contents:", items[:10])
            dataset_path = tmp_dir

    # --- Individual Upload ---
    else:
        uploaded_files = st.file_uploader(
            "Upload image and/or CSV files:",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "csv"]
        )
        if uploaded_files:
            tmp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                file_path = os.path.join(tmp_dir, f.name)
                with open(file_path, "wb") as out:
                    out.write(f.read())
            st.success(f"✅ Saved {len(uploaded_files)} files to temporary folder.")
            dataset_path = tmp_dir

    # --- Once dataset is uploaded ---
    if dataset_path and st.button("▶️ Analyze Dataset"):
        with st.spinner("Running data loader and metadata extraction..."):
            try:
                # Load dataset (auto structured if subfolders exist)
                structured = any(os.path.isdir(os.path.join(dataset_path, d)) for d in os.listdir(dataset_path))
                records = load_dataset(dataset_path, structured=structured)

                # ---- 🧩 Semi-supervised Labeling ----
                st.subheader("🏷️ Semi-Supervised Labeling")

                image_paths = [r['path'] for r in records if os.path.isfile(r['path'])]
                partial_labels = [r.get('label', None) for r in records]

                # Detect unlabeled samples
                unlabeled_count = sum(pd.isnull(partial_labels))
                if unlabeled_count == 0:
                    st.info("✅ All samples already labeled. Skipping semi-supervised labeling.")
                else:
                    st.write(f"Detected {unlabeled_count} unlabeled samples. Running label propagation...")

                    backbone_choice = st.selectbox("Select backbone model:", ["resnet18", "clip"], index=0)
                    k_neighbors = st.slider("Number of neighbors (k):", 1, 10, 5)
                    confidence_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.7, 0.05)

                    if st.button("Run Label Propagation"):
                        with st.spinner("Extracting embeddings and propagating labels..."):
                            results_df, embeddings = semi_supervised_labeling(
                                image_paths,
                                partial_labels,
                                backbone=backbone_choice,
                                k=k_neighbors,
                                conf_threshold=confidence_threshold
                            )
                            st.success("✅ Label propagation completed.")

                            st.subheader("📋 Labeled Dataset Preview")
                            st.dataframe(results_df.head(10))

                            # --- Downloadable CSV ---
                            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Download Labeled Dataset CSV",
                                data=csv_bytes,
                                file_name="semi_supervised_labels.csv",
                                mime="text/csv"
                            )


                # Extract metadata
                records = bulk_extract_metadata(records, compute_histogram=False, show_progress=False)

                # Summarize dataset
                summary = summarize_dataset(records)
                st.success("✅ Dataset loaded successfully!")

                # ---- 📊 Dataset Summary ----
                st.subheader("📊 Summary")
                st.json(summary)

                # ---- 🖼️ Sample Image Grid ----
                st.subheader("🖼️ Image Preview Grid")
                image_files = [r['path'] for r in records if r['path'].lower().endswith(('.png', '.jpg', '.jpeg'))][:12]
                cols = st.columns(4)
                for i, img_path in enumerate(image_files):
                    try:
                        with Image.open(img_path) as img:
                            cols[i % 4].image(img, width='stretch')
                    except Exception:
                        pass

                # ---- 🧾 CSV Summary (if any CSV exists) ----
                csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
                if csv_files:
                    st.subheader("📑 CSV File Summary")
                    for csv in csv_files:
                        csv_path = os.path.join(dataset_path, csv)
                        df = pd.read_csv(csv_path)
                        st.write(f"**{csv}** — shape: {df.shape}")
                        st.dataframe(df.head())
                        st.write("Column Summary:")
                        st.json({
                            "columns": list(df.columns),
                            "missing_values": df.isnull().sum().to_dict()
                        })
                        if "label" in df.columns:
                            st.bar_chart(df["label"].value_counts())

                # ---- Sample Records Table ----
                st.subheader("📋 Sample Records")
                df = pd.DataFrame([{
                    'id': r['id'],
                    'label': r.get('label'),
                    'width': r['metadata'].get('width'),
                    'height': r['metadata'].get('height'),
                    'mean_brightness': r['metadata'].get('mean_brightness'),
                    'is_corrupt': r['metadata'].get('is_corrupt')
                } for r in records[:15]])
                st.dataframe(df)

                # ---- 📊 Dataset Visualizations ----
                with st.expander("📊 Dataset Visualizations"):
                    st.pyplot(viz.plot_class_distribution(records))
                    st.pyplot(viz.plot_brightness_histogram(records))
                    st.pyplot(viz.plot_resolution_scatter(records))
                    st.pyplot(viz.plot_corruption_pie(records))
                    st.pyplot(viz.plot_label_brightness_violin(records))

            except Exception as e:
                st.error(f"❌ Error processing dataset: {e}")
            finally:
                # Cleanup
                try:
                    shutil.rmtree(dataset_path, ignore_errors=True)
                    st.info("🧹 Temporary files cleaned up successfully.")
                except Exception as cleanup_err:
                    st.warning(f"⚠️ Could not delete temp folder: {cleanup_err}")

# ---- Mode 2: Diagnostics ----
elif mode == "Run Diagnostics":
    st.header("📊 Run Diagnostics on Preprocessed Data")

    st.subheader("📦 Step 1: Extract Embeddings")

    uploaded_file = st.file_uploader("📂 Upload your dataset ZIP file", type=["zip"])

    if uploaded_file is not None:
        extract_dir = "uploaded_data_mode2"
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        st.success(f"✅ Dataset extracted to: {extract_dir}")
        dataset_path = extract_dir
    else:
        st.warning("Please upload a dataset ZIP file to continue.")
        st.stop()

    backbone_choice = st.selectbox("Select backbone model:", ["resnet18", "clip"], index=0)

    if st.button("Run Embedding Extraction"):
        with st.spinner(f"Extracting embeddings using {backbone_choice}..."):
            try:
                # Collect image paths
                image_paths = []
                for root, _, files in os.walk(dataset_path):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            image_paths.append(os.path.join(root, f))
                
                if not image_paths:
                    st.error("No images found in the provided dataset path.")
                else:
                    # Extract embeddings
                    extractor = EmbeddingExtractor(backbone=backbone_choice)
                    embeddings, ids = extractor.extract_embeddings(image_paths)
                    extractor.save_embeddings(embeddings, ids, output_dir="outputs")

                    st.success(f"✅ Extracted {embeddings.shape[0]} embeddings ({embeddings.shape[1]} dims).")
                    st.write("Embeddings saved to: `outputs/`")

                    # Store for session use
                    st.session_state.embeddings = embeddings
                    st.session_state.ids = ids
            except Exception as e:
                st.error(f"❌ Embedding extraction failed: {e}")

    if "embeddings" in st.session_state:
        st.subheader("⚡ Step 2: Build FAISS Index / Detect Duplicates")

        if st.button("Build FAISS Index"):
            with st.spinner("Building FAISS index..."):
                try:
                    index = build_faiss_index_auto(st.session_state.embeddings)
                    st.session_state.index = index
                    st.success("✅ FAISS index built successfully!")
                except Exception as e:
                    st.error(f"❌ Index building failed: {e}")

        if st.button("Find Duplicates (similarity > 0.95)"):
            with st.spinner("Running duplicate detection..."):
                try:
                    duplicates = find_duplicates_faiss_fast(st.session_state.embeddings, threshold=0.95)
                    if not duplicates:
                        st.info("✅ No significant duplicates found.")
                    else:
                        st.warning(f"⚠️ Found {len(duplicates)} potential duplicate clusters.")
                        st.write(duplicates[:10])
                except Exception as e:
                    st.error(f"❌ Duplicate detection failed: {e}")

    if "embeddings" in st.session_state:
        st.subheader("🧭 Step 3: Visualize Embeddings")

        method = st.selectbox("Select dimensionality reduction method:", ["umap", "pca", "tsne"])
        color_by = st.text_input("Color by (optional label file path):")

        if st.button("Visualize Embeddings"):
            with st.spinner(f"Reducing embeddings using {method}..."):
                try:
                    reduced = reduce_embeddings(st.session_state.embeddings, method=method)
                    fig = plot_embedding_scatter(reduced, title=f"{method.upper()} Projection")
                    st.plotly_chart(fig, use_container_width=True)

                    heatmap_fig = plot_similarity_heatmap(st.session_state.embeddings)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Visualization failed: {e}")

    if os.path.exists(output_dir):
        st.subheader("📁 Loaded Data from testing/output/")
        try:
            # --- Load embeddings ---
            embeddings_path = os.path.join(output_dir, "embeddings_resnet18.npy")
            embeddings = np.load(embeddings_path, allow_pickle=True)
            st.success(f"✅ Embeddings loaded. Shape: {embeddings.shape}")

            # --- Embedding statistics ---
            st.markdown("### 🔍 Embedding Statistics")
            stats = {
                "Min": float(np.min(embeddings)),
                "Max": float(np.max(embeddings)),
                "Mean": float(np.mean(embeddings)),
                "Std": float(np.std(embeddings)),
                "Variance": float(np.var(embeddings))
            }
            st.json(stats)

            # --- Labels ---
            labels_df = pd.read_csv(os.path.join(output_dir, "fashion_labels.csv"))
            st.markdown("### 🏷️ Sample Labels")
            st.dataframe(labels_df.head())

            if os.path.exists(os.path.join(output_dir, "sensitive_attr.npy")):
                sensitive = np.load(os.path.join(output_dir, "sensitive_attr.npy"))
                st.write(f"Sensitive attributes shape: {sensitive.shape}")
            else:
                st.warning("Sensitive attributes file not found.")

            # --- Label Distribution Chart ---
            if "label" in labels_df.columns:
                st.markdown("### 📊 Label Distribution")
                label_counts = labels_df["label"].value_counts().sort_index()
                st.bar_chart(label_counts)

                st.markdown("### 🧩 Embedding Visualizations")

                if embeddings is not None:
                    reduced = embviz.reduce_embeddings(embeddings, method="umap")

                    # 2D/3D projection
                    st.plotly_chart(embviz.plot_embedding_scatter(
                        reduced,
                        labels=labels_df['label'] if 'label' in labels_df else None
                    ))

                    # Similarity matrix
                    st.plotly_chart(embviz.plot_similarity_heatmap(embeddings))

                    # Additional plots
                    st.plotly_chart(embviz.plot_embedding_variance(embeddings))
                    st.plotly_chart(embviz.plot_class_balance_radar(
                        labels_df['label'] if 'label' in labels_df else None
                    ))

                    if os.path.exists(os.path.join(output_dir, "sensitive_attr.npy")):
                        sensitive = np.load(os.path.join(output_dir, "sensitive_attr.npy"))
                        st.plotly_chart(embviz.plot_embedding_correlation(embeddings, sensitive))
            else:
                st.info("No 'label' column found in CSV for label distribution.")

            # ---- 🩺 Diagnostics Section ----
            st.markdown("## 🩺 Dataset Diagnostics")

            if embeddings is not None and "label" in labels_df.columns:
                diagnostic_choice = st.selectbox(
                    "Choose a diagnostic to run:",
                    ["Select...", "Diversity", "Duplicates", "Imbalance", "Outliers"]
                )

                labels = labels_df["label"].values

                if diagnostic_choice == "Diversity":
                    st.subheader("Class Diversity & Separation")
                    try:
                        diversity_summary = compute_diversity_index(embeddings, labels)
                        st.write(diversity_summary)

                        overlap_df = compute_inter_class_overlap(embeddings, labels)
                        fig = plot_diversity_heatmap(overlap_df)
                        st.pyplot(fig)
                        # ✅ Save diagnostics report
                        os.makedirs("testing/output", exist_ok=True)
                        report = {
                            "type": "diversity",
                            "metrics": {
                                "diversity_summary": (
                                    diversity_summary.to_dict() if hasattr(diversity_summary, "to_dict") else str(diversity_summary)
                                )
                            }
                        }
                        with open("testing/output/diagnostics_report.json", "w") as f:
                            json.dump(report, f, indent=4)
                        st.success("📝 Diversity diagnostics report saved.")

                    except Exception as e:
                        st.error(f"❌ Diversity computation failed: {e}")

                elif diagnostic_choice == "Duplicates":
                    st.subheader("🧩 Duplicate Detection")

                    try:
                        # Handle JSON or DataFrame inputs gracefully
                        if isinstance(embeddings, str):
                            # If embeddings are given as a JSON string
                            try:
                                embeddings = np.array(json.loads(embeddings))
                            except Exception as e:
                                st.error(f"Failed to parse embeddings JSON: {e}")
                                st.stop()

                        elif isinstance(embeddings, pd.DataFrame):
                            embeddings = embeddings.select_dtypes(include=[np.number]).to_numpy()

                        elif isinstance(embeddings, list):
                            embeddings = np.array(embeddings, dtype=float)

                        if not isinstance(embeddings, np.ndarray):
                            raise ValueError("Embeddings must be a numpy array after conversion.")

                        # Run duplicate detection
                        clusters = find_duplicates(embeddings, threshold=0.98)
                        dup_summary = summarize_duplicates(clusters)

                        if dup_summary:
                            if isinstance(dup_summary, dict):
                                dup_summary = [dup_summary]

                            st.dataframe(pd.DataFrame(dup_summary))

                            csv_data = pd.DataFrame(dup_summary).to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Duplicate Report",
                                data=csv_data,
                                file_name="duplicates_report.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("✅ No duplicates found above the threshold.")

                    except Exception as e:
                        st.error(f"❌ Duplicate analysis failed: {e}")

                elif diagnostic_choice == "Imbalance":
                    st.subheader("Class Imbalance Analysis")
                    try:
                        imbalance_summary = summarize_imbalance(labels)
                        st.write(imbalance_summary["diversity"])
                        chart_data = imbalance_summary["class_distribution"].set_index("class")["count"]
                        st.bar_chart(chart_data)
                        os.makedirs("testing/output", exist_ok=True)
                        report = {
                            "type": "imbalance",
                            "metrics": {
                                "class_distribution": imbalance_summary["class_distribution"]
                                .set_index("class")["count"].to_dict()
                            }
                        }
                        with open("testing/output/diagnostics_report.json", "w") as f:
                            json.dump(report, f, indent=4)
                        st.success("📝 Imbalance diagnostics report saved.")

                    except Exception as e:
                        st.error(f"❌ Imbalance analysis failed: {e}")

                elif diagnostic_choice == "Outliers":
                    st.subheader("Outlier Analysis")
                    try:
                        outlier_summary = summarize_outliers(embeddings, labels)
                        st.dataframe(outlier_summary["embedding_outliers"])
                        st.line_chart(outlier_summary["knn_scores"])
                        os.makedirs("testing/output", exist_ok=True)
                        report = {
                            "type": "outliers",
                            "metrics": {
                                "num_outliers": len(outlier_summary["embedding_outliers"]),
                                "mean_knn_score": float(np.mean(outlier_summary["knn_scores"]))
                            }
                        }
                        with open("testing/output/diagnostics_report.json", "w") as f:
                            json.dump(report, f, indent=4)
                        st.success("📝 Outlier diagnostics report saved.")

                    except Exception as e:
                        st.error(f"❌ Outlier analysis failed: {e}")
            else:
                st.info("Please ensure both embeddings and labels are loaded to run diagnostics.")

            st.header("🛠️ Dataset Repair Suggestions")

            if st.button("Generate Repair Suggestions"):
                suggestions = generate_repair_suggestions("testing/output/diagnostics_report.json")
                if "error" in suggestions:
                    st.error(suggestions["error"])
                else:
                    for s in suggestions["suggestions"]:
                        st.write(f"- {s}")

                if st.button("Apply Fixes"):
                    repaired_path = apply_repairs(df, label_col="label")
                    st.success(f"Repaired dataset saved: {repaired_path}")


            st.markdown("---")
            st.header("🔍 Influence & Fairness Analysis")

            try:
                # Ensure sensitive attributes exist
                if not os.path.exists(SENSITIVE_ATTR_PATH):
                    st.warning("Sensitive attribute file not found. Generating one automatically...")
                    create_sensitive_attribute_file()

                sensitive_attr = np.load(SENSITIVE_ATTR_PATH, allow_pickle=True)

                # Load embeddings and labels
                embeddings_path = os.path.join(output_dir, "embeddings_resnet18.npy")
                labels_path = os.path.join(output_dir, "labels.npy")

                if os.path.exists(embeddings_path) and os.path.exists(labels_path):
                    embeddings = np.load(embeddings_path)
                    labels = np.load(labels_path)

                    st.write("✅ Loaded embeddings and labels successfully.")

                    # Compute influence scores
                    st.write("Computing influence scores...")
                    influence_scores = compute_influence_scores(embeddings, labels)
                    st.success("Influence scores computed successfully!")

                    # Bias-conflicting samples
                    bias_df = identify_bias_conflicting_samples(embeddings, labels, sensitive_attr)
                    st.dataframe(bias_df.head())

                    # Fairness metrics
                    st.write("Computing fairness metrics...")
                    fairness_metrics = evaluate_fairness_metrics(labels, labels, sensitive_attr)
                    st.json(fairness_metrics)

                    # Download option
                    csv_path = os.path.join(output_dir, "influence_results.csv")
                    bias_df.to_csv(csv_path, index=False)
                    st.download_button("⬇️ Download Influence Results", csv_path)

                else:
                    st.error("Embeddings or labels not found in output directory.")

            except Exception as e:
                st.error(f"❌ Influence analysis failed: {e}")

            # # --- Auxiliary output files ---
            # st.markdown("### 📁 Other Available Outputs")
            # files = [
            #     "faiss.index", "image_ids.txt",
            #     "knn_graph.npy", "labels.npy",
            #     "train_labels.csv", "val_labels.csv"
            # ]
            # available = [f for f in files if os.path.exists(os.path.join(output_dir, f))]

            # if available:
            #     st.success("Detected additional files:")
            #     for f in available:
            #         file_path = os.path.join(output_dir, f)
            #         file_ext = os.path.splitext(f)[1].lower()

            #         with open(file_path, "rb") as data_file:
            #             st.download_button(
            #                 label=f"⬇️ Download {f}",
            #                 data=data_file,
            #                 file_name=f,
            #                 mime="application/octet-stream"
            #             )

            #         # Inline previews for readable file types
            #         try:
            #             if file_ext == ".csv":
            #                 df = pd.read_csv(file_path)
            #                 st.write(f"**Preview of {f}** — shape: {df.shape}")
            #                 st.dataframe(df.head())

            #             elif file_ext == ".txt":
            #                 with open(file_path, "r", encoding="utf-8", errors="ignore") as txt_file:
            #                     content = txt_file.read(500)
            #                     st.text_area(f"Preview of {f}", content, height=150)

            #             elif file_ext == ".npy":
            #                 arr = np.load(file_path)
            #                 st.write(f"**{f}** — shape: {arr.shape}, dtype: {arr.dtype}")
            #                 st.write("Sample values:")
            #                 st.write(arr[:min(5, len(arr))])
            #         except Exception as preview_err:
            #             st.warning(f"⚠️ Could not preview {f}: {preview_err}")
            # else:
            #     st.info("No auxiliary output files found.")

        except Exception as e:
            st.error(f"❌ Error loading or visualizing diagnostics data: {e}")

        finally:
            tmp_dir = os.path.join(output_dir, "temp")
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    st.info("🧹 Temporary diagnostics cache cleaned up.")
                except Exception as cleanup_err:
                    st.warning(f"⚠️ Could not delete diagnostics temp folder: {cleanup_err}")

    else:
        st.warning("⚠️ Output folder not found. Please run Task 1 first.")
