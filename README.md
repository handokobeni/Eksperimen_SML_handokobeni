# Eksperimen SML — Heart Failure Prediction

Kriteria 1 Proyek Akhir MSML (target Advance).

## Struktur
```
Eksperimen_SML_handokobeni/
├── .github/workflows/preprocessing.yml   # [Advance] automasi preprocessing
├── heart_raw.csv                          # dataset mentah
├── requirements.txt
└── preprocessing/
    ├── Eksperimen_handokobeni.ipynb       # notebook (pakai Template MSML)
    ├── automate_handokobeni.py            # [Skilled] preprocessing otomatis
    ├── heart_preprocessing.csv            # output (dihasilkan)
    └── preprocessor.joblib                # pipeline tersimpan (dihasilkan)
```

## Dataset
Heart Failure Prediction — klasifikasi biner (`HeartDisease`). 918 baris, 11 fitur (numerik + kategorikal).

## Cara jalan lokal
```bash
pip install -r requirements.txt
python preprocessing/automate_handokobeni.py heart_raw.csv preprocessing/heart_preprocessing.csv
```
