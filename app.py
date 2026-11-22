from pathlib import Path
import logging
import pickle

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "Frontend"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "loan_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

FEATURES = [
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "existing_debt",
    "debt_to_income_ratio",
    "employment_years",
    "previous_defaults",
]


def load_scaler(path):
    """Load a scaler saved with joblib or pickle."""
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as file:
                return pickle.load(file)
        except Exception as exc:
            raise RuntimeError(
                "Could not load scaler.pkl. Ensure it is a valid fitted scaler "
                "saved with pickle/joblib and compatible with loan_model.keras."
            ) from exc


if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not SCALER_PATH.is_file():
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
app.logger.info("Keras model and scaler loaded.")

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/health")
def health():
    return jsonify({
        "success": True,
        "status": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Send application details as a JSON object."}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": "Some required fields are missing.",
                "missing_features": missing
            }), 400

        values = {}
        for feature in FEATURES:
            try:
                value = float(data[feature])
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": f"'{feature}' must be a valid number."}), 400
            if not np.isfinite(value):
                return jsonify({"success": False, "error": f"'{feature}' must be finite."}), 400
            values[feature] = value

        if values["annual_income"] <= 0:
            return jsonify({"success": False, "error": "Annual income must be greater than zero to calculate the ratio."}), 400
        if values["loan_amount"] < 0 or values["existing_debt"] < 0:
            return jsonify({"success": False, "error": "Loan amount and existing debt cannot be negative."}), 400
        if values["loan_term_months"] <= 0:
            return jsonify({"success": False, "error": "Loan term must be greater than zero."}), 400
        if values["credit_score"] < 0 or values["employment_years"] < 0 or values["previous_defaults"] < 0:
            return jsonify({"success": False, "error": "Credit score, employment years, and previous defaults cannot be negative."}), 400

        # Recalculate the ratio on the server to avoid stale client values.
        # Assumption: existing_debt represents an annual debt amount.
        values["debt_to_income_ratio"] = values["existing_debt"] / values["annual_income"]

        input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        scaled_input = scaler.transform(input_df)
        raw_prediction = model.predict(scaled_input, verbose=0)
        probability = float(np.asarray(raw_prediction).reshape(-1)[0])

        if not np.isfinite(probability) or probability < 0 or probability > 1:
            raise ValueError("Model output is not a valid probability.")

        approved = probability >= 0.5
        status = "Approved" if approved else "Rejected"

        return jsonify({
            "success": True,
            "status": status,
            "prediction": status,
            "approved_probability": round(probability * 100, 2),
            "rejected_probability": round((1 - probability) * 100, 2),
            "debt_to_income_ratio": round(values["debt_to_income_ratio"], 4),
            "message": "The model predicts approval." if approved else "The model predicts rejection."
        })
    except Exception:
        app.logger.exception("Loan prediction failed.")
        return jsonify({
            "success": False,
            "error": "Prediction failed. Check the server terminal and model/scaler compatibility."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
