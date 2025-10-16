# src/generate_plots.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "default_of_credit_card_clients.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "logistic_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.joblib")
STATIC_DIR = os.path.join(BASE_DIR, "..", "app", "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")

# ===== Buat folder output =====
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===== Load data =====
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.rename(columns={"default.payment.next.month": "default"})
df['default'] = df['default'].astype(str).str.strip().astype(int)

# ===== EDA Plots =====
plt.figure(figsize=(6, 4))
sns.countplot(x="default", data=df)
plt.title("Distribusi Default")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "generated_1.png"))
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(df["LIMIT_BAL"], bins=50, kde=True)
plt.title("Distribusi LIMIT_BAL")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "generated_2.png"))
plt.close()

plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Heatmap Korelasi Fitur")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
plt.close()

# ===== Load model & scaler =====
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===== Siapkan data untuk evaluasi =====
X = df.drop(columns=["ID","default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ===== Confusion Matrix =====
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Default", "Default"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.close()

# ===== ROC Curve =====
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="orange", label=f"ROC (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
plt.close()

print(f"✅ Semua plot berhasil dibuat di folder:\n{IMAGES_DIR}\n{PLOTS_DIR}")
