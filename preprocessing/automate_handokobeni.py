"""
Kriteria 1 (Skilled/Advance) — Automasi Preprocessing
Dataset: Heart Failure Prediction (target: HeartDisease)

Konversi dari notebook eksperimen menjadi fungsi otomatis yang mengembalikan
data siap latih + menyimpan pipeline preprocessing (joblib) untuk konsistensi
saat training & inference.

Jalankan:
    python automate_handokobeni.py <raw_csv> <output_csv>
Contoh:
    python automate_handokobeni.py ../heart_raw.csv heart_preprocessing.csv
"""
import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

TARGET = "HeartDisease"
RANDOM_STATE = 42


def build_preprocessor(X):
    """Bangun ColumnTransformer berdasarkan tipe kolom (numerik vs kategorikal)."""
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_tf, numeric),
        ("cat", categorical_tf, categorical),
    ])


def preprocess_data(df, target_column=TARGET, pipeline_path="preprocessor.joblib"):
    """Fit preprocessing pada train, transform train & test, simpan pipeline.

    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_t = preprocessor.fit_transform(X_train)   # fit HANYA di train (hindari leakage)
    X_test_t = preprocessor.transform(X_test)

    dump(preprocessor, pipeline_path)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return X_train_t, X_test_t, y_train, y_test, feature_names


def main():
    raw_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "heart_raw.csv")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "heart_preprocessing.csv"
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(out_dir, exist_ok=True)
    pipeline_path = os.path.join(out_dir, "preprocessor.joblib")

    df = pd.read_csv(raw_csv)
    X_train, X_test, y_train, y_test, feats = preprocess_data(
        df, TARGET, pipeline_path=pipeline_path
    )

    train_df = pd.DataFrame(X_train, columns=feats)
    train_df[TARGET] = y_train.values
    test_df = pd.DataFrame(X_test, columns=feats)
    test_df[TARGET] = y_test.values
    full = pd.concat([train_df, test_df], ignore_index=True)
    full.to_csv(out_csv, index=False)

    print(f"[OK] Pipeline    -> {pipeline_path}")
    print(f"[OK] Data siap   -> {out_csv}  shape={full.shape}")
    print(f"[OK] Fitur ({len(feats)}): {feats}")


# self-check ringan — pastikan output punya kolom target & tak kosong
def demo():
    import numpy as np
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "Age": rng.randint(30, 70, 50),
        "Sex": rng.choice(["M", "F"], 50),
        "Cholesterol": rng.randint(100, 300, 50).astype(float),
        "HeartDisease": rng.randint(0, 2, 50),
    })
    Xtr, Xte, ytr, yte, feats = preprocess_data(df, pipeline_path="/tmp/_pp.joblib")
    assert Xtr.shape[0] == 40 and Xte.shape[0] == 10
    assert len(feats) >= 3
    print("demo OK", Xtr.shape, feats)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
