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

# Numeric features selected from dataset.csv
NUMERIC_FEATURES = [
    "Age (Years)",
    "Weight (kg)",
    "Height (cm)",
    "Top_Speed (kmh)",
]

# Categorical mapping matching dataset.csv regions
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

    # Step 1: Read CSV and drop rows with missing values
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    # Step 2: Combine numeric and categorical feature columns
    all_feature_cols = NUMERIC_FEATURES + list(TEXT_FEATURES.keys())
    X_df = df[all_feature_cols].copy()

    # Map text values to numbers
    for col, mapping in TEXT_FEATURES.items():
        X_df[col] = df[col].map(mapping)

    # Map target strings to integer labels
    class_names = sorted(df[TARGET].unique())
    label_mapping = {name: i for i, name in enumerate(class_names)}
    y_series = df[TARGET].map(label_mapping)

    # Filter out classes with < 3 samples so 3-way stratified splitting works
    class_counts = y_series.value_counts()
    valid_classes = class_counts[class_counts >= 3].index

    mask = y_series.isin(valid_classes)
    X_df = X_df[mask]
    y_series = y_series[mask]

    # Keep only class names corresponding to filtered dataset
    class_names = [name for name in class_names if label_mapping[name] in valid_classes]

    X = X_df.to_numpy(dtype="float32")
    y = y_series.to_numpy(dtype="int32")

    # Step 3: Stratified split into Train (60%) / Validation (20%) / Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )  # 0.25 x 0.8 = 0.2 of overall dataset

    # Step 4: Feature Scaling using StandardScaler
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

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("Train shape :", data["X_train"].shape)
    print("Val shape   :", data["X_val"].shape)
    print("Test shape  :", data["X_test"].shape)
    print("Classes     :", data["class_names"])
