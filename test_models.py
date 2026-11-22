from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "loan_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

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


def main():
    print("Checking Loan Approval model files...\n")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")

    model = load_model(MODEL_PATH)
    print("SUCCESS: Keras model loaded.")

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    print("SUCCESS: Scaler loaded.")

    sample = pd.DataFrame(
        [[60000, 700, 15000, 36, 5000, 0.25, 5, 0]],
        columns=FEATURES
    )

    scaled_sample = scaler.transform(sample)
    prediction = model.predict(scaled_sample, verbose=0)
    probability = float(np.asarray(prediction).reshape(-1)[0])

    print("\nSUCCESS: Test prediction completed.")
    print(f"Model output: {probability:.4f}")
    print(f"Predicted approval probability: {probability * 100:.2f}%")
    print("Predicted class:", "Approved" if probability >= 0.5 else "Rejected")
    print("\nBoth files appear loadable and usable together.")
    print("Note: this test does not establish prediction accuracy.")


if __name__ == "__main__":
    main()
