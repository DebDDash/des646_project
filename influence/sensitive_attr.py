import numpy as np
import os
import pandas as pd

PROJECT_ROOT = os.getcwd()
LABELS_PATH = os.path.join(PROJECT_ROOT, "testing", "output", "labels.npy")
CSV_PATHS = [
    os.path.join(PROJECT_ROOT, "testing", "output", "labels.csv"),
    os.path.join(PROJECT_ROOT, "testing", "output", "train_labels.csv"),
    os.path.join(PROJECT_ROOT, "testing", "output", "val_labels.csv"),
]

SENSITIVE_ATTR_DIR = os.path.join(PROJECT_ROOT, "influence")
SENSITIVE_ATTR_PATH = os.path.join(SENSITIVE_ATTR_DIR, "sensitive_attr.npy")

# Common sensitive column keywords
SENSITIVE_COLUMNS = ["gender", "sex", "age", "group", "race", "region"]

def create_sensitive_attribute_file():
    # Determine number of samples
    try:
        labels = np.load(LABELS_PATH)
        N = len(labels)
        print(f"Found {N} samples from {os.path.basename(LABELS_PATH)}.")
    except FileNotFoundError:
        print(f"🛑 labels.npy not found — using default N=200.")
        N = 200

    sensitive_attr = None

    # Try to infer sensitive attributes from CSVs
    for csv_path in CSV_PATHS:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            found = [col for col in df.columns if any(key in col.lower() for key in SENSITIVE_COLUMNS)]
            if found:
                col = found[0]
                print(f"✅ Using '{col}' from {os.path.basename(csv_path)} as sensitive attribute.")
                sensitive_attr = df[col].values
                break

    # Fallback: random attribute
    if sensitive_attr is None:
        print("⚠️ No sensitive column found — generating random Group A/B attributes.")
        np.random.seed(42)
        sensitive_attr = np.random.choice(["Group A", "Group B"], size=N)

    os.makedirs(SENSITIVE_ATTR_DIR, exist_ok=True)
    np.save(SENSITIVE_ATTR_PATH, sensitive_attr, allow_pickle=True)
    print(f"\n✅ Created sensitive_attr.npy with {len(sensitive_attr)} samples at {SENSITIVE_ATTR_PATH}")

if __name__ == "__main__":
    create_sensitive_attribute_file()

