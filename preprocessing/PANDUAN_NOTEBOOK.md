# Panduan Isi Notebook Eksperimen (Kriteria 1 — Basic)

> ⚠️ Notebook **wajib** memakai **Template Eksperimen MSML** resmi dari Dicoding.
> Unduh template itu, lalu isi tiap section dengan blok kode di bawah.
> Nama notebook: `Eksperimen_handokobeni.ipynb` (letakkan di folder `preprocessing/`).
> Dataset mentah: `heart_raw.csv` (letakkan di root repo).

## Cell 1 — Import
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Cell 2 — Data Loading
```python
df = pd.read_csv("../heart_raw.csv")
df.head()
```
```python
df.info()
df.describe()
df.isnull().sum()
```

## Cell 3 — EDA (wajib manual, tak diotomasi)
```python
# Distribusi target
sns.countplot(x="HeartDisease", data=df); plt.title("Distribusi HeartDisease"); plt.show()

# Histogram fitur numerik
df.select_dtypes(include="number").hist(figsize=(12, 8)); plt.tight_layout(); plt.show()

# Boxplot deteksi outlier (contoh)
sns.boxplot(x=df["Cholesterol"]); plt.title("Boxplot Cholesterol"); plt.show()

# Korelasi fitur numerik
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix"); plt.show()
```
> Tulis 2–3 kalimat insight tiap visual (mis. kelas seimbang/tidak, ada outlier di Cholesterol=0, fitur mana berkorelasi dengan target).

## Cell 4 — Preprocessing manual
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Pisah fitur & target
X = df.drop(columns=["HeartDisease"]); y = df["HeartDisease"]
numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(include="object").columns.tolist()

# Imputasi + scaling numerik
X[numeric] = SimpleImputer(strategy="median").fit_transform(X[numeric])
X[numeric] = StandardScaler().fit_transform(X[numeric])

# Encoding kategorikal
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = enc.fit_transform(X[categorical])
X_enc = pd.concat(
    [X[numeric].reset_index(drop=True),
     pd.DataFrame(encoded, columns=enc.get_feature_names_out(categorical))],
    axis=1,
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)
```

## Cell 5 — Simpan hasil (opsional untuk verifikasi)
```python
X_enc.assign(HeartDisease=y).to_csv("heart_preprocessing.csv", index=False)
```

---
### Langkah Skilled/Advance
Setelah eksperimen manual di atas jalan, **semua langkah preprocessing sudah dikonversi** ke
`automate_handokobeni.py` (di folder ini). Uji lokal:
```bash
python automate_handokobeni.py ../heart_raw.csv heart_preprocessing.csv
```
Advance: GitHub Actions (`.github/workflows/preprocessing.yml`) menjalankan script itu otomatis tiap push.
