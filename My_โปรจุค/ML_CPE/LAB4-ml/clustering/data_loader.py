import sys
sys.dont_write_bytecode = True  # Prevent creating __pycache__ directory

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "Data_Set" / "dataset.csv"

# Numerical features present in dataset.csv used for clustering
FEATURES = [
    "Age (Years)",
    "Weight (kg)",
    "Height (cm)",
    "Price_THB",
    "Top_Speed (kmh)",
]

# ---------------------------------------------------------------------------
def load_data():
    """
    Returns a dictionary containing:
        X        : Standardized feature matrix (mean=0, std=1) for clustering models
        X_raw    : Unscaled raw values (used for interpreting cluster profiles)
        df       : Full pandas DataFrame loaded from CSV
        features : List of feature column names used
    """
    df = pd.read_csv(CSV_PATH).dropna()

    X_raw = df[FEATURES].to_numpy(dtype="float32")
    X = StandardScaler().fit_transform(X_raw).astype("float32")  # Mean=0, Std=1

    return {
        "X": X,
        "X_raw": X_raw,
        "df": df,
        "features": FEATURES,
    }

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("Data shape                             :", data["X"].shape)
    print("Mean after scale (should be close to 0) :", data["X"].mean(axis=0).round(3))
    print("Std after scale  (should be close to 1) :", data["X"].std(axis=0).round(3))
