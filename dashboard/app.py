import os
import sys
import shutil
import tempfile
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add parent directory to Python path (so we can import data_utils)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.loader import load_dataset, bulk_extract_metadata, summarize_dataset
from data_utils import visualize as viz
from embedding import visualize as embviz

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

    if os.path.exists(output_dir):
        st.subheader("📁 Loaded Data from testing/output/")
        try:
            # --- Load embeddings ---
            embeddings_path = os.path.join(output_dir, "embeddings_resnet18.npy")
            embeddings = np.load(embeddings_path)
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

            # --- Auxiliary output files ---
            st.markdown("### 📁 Other Available Outputs")
            files = [
                "faiss.index", "image_ids.txt",
                "knn_graph.npy", "labels.npy",
                "train_labels.csv", "val_labels.csv"
            ]
            available = [f for f in files if os.path.exists(os.path.join(output_dir, f))]

            if available:
                st.success("Detected additional files:")
                for f in available:
                    file_path = os.path.join(output_dir, f)
                    file_ext = os.path.splitext(f)[1].lower()

                    with open(file_path, "rb") as data_file:
                        st.download_button(
                            label=f"⬇️ Download {f}",
                            data=data_file,
                            file_name=f,
                            mime="application/octet-stream"
                        )

                    # Inline previews for readable file types
                    try:
                        if file_ext == ".csv":
                            df = pd.read_csv(file_path)
                            st.write(f"**Preview of {f}** — shape: {df.shape}")
                            st.dataframe(df.head())

                        elif file_ext == ".txt":
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as txt_file:
                                content = txt_file.read(500)
                                st.text_area(f"Preview of {f}", content, height=150)

                        elif file_ext == ".npy":
                            arr = np.load(file_path)
                            st.write(f"**{f}** — shape: {arr.shape}, dtype: {arr.dtype}")
                            st.write("Sample values:")
                            st.write(arr[:min(5, len(arr))])
                    except Exception as preview_err:
                        st.warning(f"⚠️ Could not preview {f}: {preview_err}")
            else:
                st.info("No auxiliary output files found.")

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
