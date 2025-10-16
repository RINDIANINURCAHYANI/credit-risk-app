# src/utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, target_col, use_smote=False):
    # contoh: pilih beberapa fitur numerik dasar (ubah sesuai dataset)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # if any categorical columns, encode them here (one-hot or label)
    # For simplicity, assume all numeric or already encoded
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, scaler
