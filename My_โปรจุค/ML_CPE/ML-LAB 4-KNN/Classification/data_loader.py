"""
Read CSV dataset.csv
Convert categorical text to numeric values
Apply StandardScaler for KNN classification
Split data: Train (60%) / Validation (20%) / Test (20%)
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "Data_Set" / "dataset.csv"

TARGET = "Type_car"

NUMERIC_FEATURES = [
    "Age (Years)",
    "Weight (kg)",
    "Height (cm)",
    "Top_Speed (kmh)",
]

TEXT_FEATURES = {
    "Primary_Region": {
        "Japan": 0,
        "USA": 1,
        "Germany": 2,
        "South Korea": 3,
        "China": 4,
        "Sweden": 5,
        "Italy": 6,
        "UK": 7,
    },
}

# ---------------------------------------------------------------------------
def load_data(test_size=0.2, seed=42):

    # Step 1: Read CSV and drop missing values
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    # Step 2: Combine numeric and categorical features
    all_feature_cols = NUMERIC_FEATURES + list(TEXT_FEATURES.keys())
    X_df = df[all_feature_cols].copy()

    for col, mapping in TEXT_FEATURES.items():
        X_df[col] = df[col].map(mapping)

    # Filter out classes with < 3 samples
    class_counts = df[TARGET].value_counts()
    valid_class_names = class_counts[class_counts >= 3].index

    # Keep only valid classes
    mask = df[TARGET].isin(valid_class_names)
    X_df = X_df[mask]
    df_filtered = df[mask]

    # Re-index target labels cleanly (0, 1, 2, 3...) after filtering
    class_names = sorted(df_filtered[TARGET].unique())
    clean_label_mapping = {name: i for i, name in enumerate(class_names)}
    y_series = df_filtered[TARGET].map(clean_label_mapping)

    X = X_df.to_numpy(dtype="float32")
    y = y_series.to_numpy(dtype="int32")

    # Step 3: Stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )

    # Step 4: Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": all_feature_cols,
        "n_rows": len(X),
    }


if __name__ == "__main__":
    data = load_data()
    print("Train shape :", data["X_train"].shape)
    print("Classes     :", data["class_names"])
