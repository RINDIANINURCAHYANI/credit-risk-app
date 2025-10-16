# app/app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Path model & scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/logistic_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Sesuaikan dengan 23 fitur yang dipakai saat training
FEATURES = [
    "LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"
]

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        vals = []
        for f in FEATURES:
            v = request.form.get(f, type=float)
            vals.append(v if v is not None else 0.0)
        X = np.array(vals).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0,1]
        label = "Berisiko Gagal Bayar" if prob >= 0.5 else "Tidak Berisiko"

        return render_template("result.html",
                               prob=round(float(prob),4),
                               label=label,
                               images=[
                                   "/static/images/generated_1.png",
                                   "/static/images/generated_2.png"
                               ],
                               plots=[
                                   "/static/plots/confusion_matrix.png",
                                   "/static/plots/roc_curve.png"
                               ])
    return render_template("index.html",
                           images=[
                               "/static/images/generated_1.png",
                               "/static/images/generated_2.png"
                           ])

if __name__ == "__main__":
    app.run(debug=True)
