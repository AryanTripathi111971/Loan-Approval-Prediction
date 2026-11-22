# Loan Approval Prediction using Artificial Neural Network (ANN)

A full-stack loan approval prediction demo that combines a trained Keras neural network with a Flask API and a responsive HTML, CSS, and JavaScript interface.

> **Disclaimer:** This project is an educational demonstration. Its output is a model estimate, not a real lending decision or financial advice.

> **Setup status:** `models/scaler.pkl` is in this repository. The supplied ZIP also contains `models/loan_model.keras`, but that binary model file has not yet been uploaded here. Add it to the `models/` folder before running `app.py` or `test_models.py`.

## Overview

LoanWise AI collects eight applicant attributes and sends them to a Flask prediction endpoint. The backend validates the input, calculates the debt-to-income ratio, applies the saved feature scaler, and passes the transformed data to the trained neural network.

The model returns an approval probability. The application displays the estimated approval/rejection result and both probabilities.

## Features

- ANN model loaded from a Keras `.keras` file
- Saved feature scaler loaded with joblib/pickle
- Flask backend with `/predict`, `/api/predict`, and `/health` endpoints
- Browser-based applicant form
- Automatic debt-to-income ratio calculation
- Input validation and readable error responses
- Approval and rejection probability display

## Tech Stack

- **Language:** Python, JavaScript
- **Machine Learning:** TensorFlow / Keras, NumPy, pandas, scikit-learn-compatible scaler
- **Backend:** Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Model artifacts:** `loan_model.keras`, `scaler.pkl`

## Project Structure

```text
Loan-Approval-Prediction/
├── app.py
├── test_models.py
├── requirements.txt
├── .gitignore
├── README.md
├── Frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
└── models/
    ├── loan_model.keras
    └── scaler.pkl
```

## Input Features

The backend expects these eight numeric features in this exact order:

| Feature | Description |
|---|---|
| `annual_income` | Applicant's annual income |
| `credit_score` | Applicant's credit score |
| `loan_amount` | Requested loan amount |
| `loan_term_months` | Loan duration in months |
| `existing_debt` | Existing debt amount |
| `debt_to_income_ratio` | Existing debt divided by annual income |
| `employment_years` | Years of employment |
| `previous_defaults` | Number of previous defaults |

The current implementation calculates `debt_to_income_ratio = existing_debt / annual_income` on both the client and server. This assumes that `existing_debt` is expressed on an annual basis. If the training data represents monthly debt payments, the formula must be adjusted to match the training preprocessing.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AryanTripathi111971/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the model artifact check

```bash
python test_models.py
```

This script checks that the saved model and scaler can be loaded and performs one sample inference. It does **not** measure model accuracy or validate performance on a test dataset.

### 5. Start the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## API Endpoints

### `GET /health`

Returns application status and whether the model and scaler have loaded.

### `POST /predict` or `POST /api/predict`

Accepts a JSON object containing all eight features.

Example request:

```json
{
  "annual_income": 60000,
  "credit_score": 700,
  "loan_amount": 15000,
  "loan_term_months": 36,
  "existing_debt": 5000,
  "debt_to_income_ratio": 0.0833,
  "employment_years": 5,
  "previous_defaults": 0
}
```

The backend recalculates the debt-to-income ratio before scaling the input. A successful response includes `status`, `approved_probability`, `rejected_probability`, and the calculated ratio.

## Important Notes

- Keep the model and scaler together in the `models/` directory.
- The scaler and neural network must be from the same training pipeline and use the feature order listed above.
- The repository currently contains the trained artifacts and inference application; the training dataset and model-training notebook/script were not included in the supplied project archive.
- No accuracy, precision, recall, F1-score, or other evaluation metric is claimed because those evaluation results were not part of the supplied files.
- Do not use this demo to make actual credit or lending decisions.

## Future Improvements

- Add a documented model-training pipeline and dataset source.
- Add a reproducible evaluation report with appropriate classification metrics.
- Add automated API tests and CI workflow.
- Add deployment configuration and production-grade error handling.

## Author

**Aryan Tripathi**  
GitHub: [@AryanTripathi111971](https://github.com/AryanTripathi111971)
