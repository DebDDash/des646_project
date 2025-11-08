# DES646 Project

# Dataset Bias and Quality Diagnostics Dashboard

## Project Description

This project presents a **lightweight, interactive Streamlit dashboard** designed to help researchers and designers **detect, visualize, and mitigate dataset bias and quality issues** before training machine learning models.  
By combining principles of **explainable AI** and **visual analytics**, it enables transparent and interpretable exploration of image datasets even for users without deep technical expertise.

The tool automatically identifies common dataset issues such as **duplicates**, **class imbalance**, **outliers**, and **potential bias-conflicting samples** using pretrained vision models. It also supports **semi-supervised labeling** for partially labeled datasets and provides **actionable feedback** for improving data quality and fairness.

Key functionalities include:
- **Data Diagnostics:** Detect duplicates, imbalance, outliers, and low-quality samples.  
- **Bias & Self-Influence Analysis:** Identify samples that disproportionately affect model fairness or accuracy.  
- **Visual Exploration:** Explore class distributions, embedding projections (UMAP), and influence-ranked images.  
- **Label Completion (Optional):** Suggest probable labels for unlabeled images using embedding similarity.  
- **Actionable Feedback:** Recommend relabeling or removing problematic samples to improve dataset quality.  

---

## How to Run

### 1. Clone the repository and navigate to the project directory
```
git clone <repo-url>
cd <repo-name>
```

### 2. Install dependencies
Make sure you have Python 3.8 or above, then run:
```
pip install -r requirements.txt
```

### 3. Launch the Streamlit dashboard
```
streamlit run dashboard/app.py
```

### 4. Use the interface
- Choose whether to upload a semi-labeled dataset or run diagnostics on an existing dataset.

- Explore duplicates, imbalance, outliers, and influence scores through interactive visualizations.

- Download reports for duplicate and imbalance analysis.



