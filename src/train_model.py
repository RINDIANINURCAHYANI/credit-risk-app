# src/train_model.py
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from utils import load_data, preprocess

# === 1. Tentukan path file CSV ===
CSV_FILENAME = "default_of_credit_card_clients.csv"  # pastikan nama file sesuai
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", CSV_FILENAME)
TARGET = "default.payment.next.month"  # sesuaikan dengan kolom target dataset

# === 2. Pastikan file CSV tersedia ===
if not os.path.exists(DATA_PATH):
    print(f"❌ File CSV tidak ditemukan di: {DATA_PATH}")
    print("📝 Pastikan nama file dan lokasinya benar.")
    sys.exit(1)

# === 3. Pastikan folder output ada ===
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "static", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# === 4. Load dan preprocessing data ===
print("📥 Membaca data...")
df = load_data(DATA_PATH)
print(f"✅ Data berhasil dibaca. Jumlah baris: {len(df)}")

# 🧹 Hapus kolom ID sejak awal supaya tidak ikut training
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# === 5. Split & scaling ===
X_train, X_test, y_train, y_test, scaler = preprocess(df, TARGET, use_smote=False)

# === 6. Latih model ===
print("🤖 Melatih model Logistic Regression...")
model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih!")

# === 7. Simpan model & scaler ===
model_path = os.path.join(MODELS_DIR, "logistic_model.joblib")
scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"💾 Model disimpan di: {model_path}")
print(f"💾 Scaler disimpan di: {scaler_path}")

# === 8. Evaluasi model ===
preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, preds))

# === 9. Plot Confusion Matrix ===
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_title("Confusion Matrix")
confusion_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(confusion_path, bbox_inches='tight')
plt.close()
print(f"📌 Confusion matrix plot disimpan di: {confusion_path}")

# === 10. Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
plt.savefig(roc_path, bbox_inches='tight')
plt.close()
print(f"📌 ROC curve plot disimpan di: {roc_path}")

print("\n✅ Training selesai tanpa error!")
