const loanForm = document.getElementById("loanForm");
const submitButton = document.getElementById("submitButton");
const formError = document.getElementById("formError");
const dtiInput = document.getElementById("debt_to_income_ratio");
const annualIncomeInput = document.getElementById("annual_income");
const existingDebtInput = document.getElementById("existing_debt");

const resultIcon = document.getElementById("resultIcon");
const resultTitle = document.getElementById("resultTitle");
const resultDescription = document.getElementById("resultDescription");
const probabilityBlock = document.getElementById("probabilityBlock");
const approvalValue = document.getElementById("approvalValue");
const rejectionValue = document.getElementById("rejectionValue");
const approvalMeter = document.getElementById("approvalMeter");
const resultMessage = document.getElementById("resultMessage");

const FEATURE_IDS = [
  "annual_income", "credit_score", "loan_amount", "loan_term_months",
  "existing_debt", "debt_to_income_ratio", "employment_years", "previous_defaults"
];

function updateDebtToIncomeRatio() {
  const annualIncome = Number(annualIncomeInput.value);
  const existingDebt = Number(existingDebtInput.value);

  if (annualIncomeInput.value === "" || existingDebtInput.value === "" ||
      !Number.isFinite(annualIncome) || !Number.isFinite(existingDebt) ||
      annualIncome <= 0 || existingDebt < 0) {
    dtiInput.value = "";
    return;
  }

  const ratio = existingDebt / annualIncome;
  dtiInput.value = ratio.toFixed(4);
}

annualIncomeInput.addEventListener("input", updateDebtToIncomeRatio);
existingDebtInput.addEventListener("input", updateDebtToIncomeRatio);

function showError(message) {
  formError.textContent = message;
  formError.classList.remove("hidden");
}
function clearError() {
  formError.textContent = "";
  formError.classList.add("hidden");
}
function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.querySelector(".button-text").textContent =
    loading ? "Analyzing..." : "Analyze application";
}

function showResult(data) {
  const approved = String(data.status || data.prediction || "").toLowerCase() === "approved";
  const approval = Number(data.approved_probability);
  const rejection = Number(data.rejected_probability);
  if (!Number.isFinite(approval) || !Number.isFinite(rejection)) {
    throw new Error("Server response did not include valid probability values.");
  }
  resultIcon.textContent = approved ? "✓" : "!";
  resultIcon.classList.toggle("approved", approved);
  resultIcon.classList.toggle("rejected", !approved);
  resultTitle.textContent = approved ? "Likely to be approved" : "Likely to be rejected";
  resultDescription.textContent = approved
    ? "The model estimates this application is more likely to be approved."
    : "The model estimates this application is more likely to be rejected.";
  approvalValue.textContent = `${approval.toFixed(2)}%`;
  rejectionValue.textContent = `${rejection.toFixed(2)}%`;
  approvalMeter.style.width = `${Math.max(0, Math.min(100, approval))}%`;
  probabilityBlock.classList.remove("hidden");
  resultMessage.textContent = data.message || "This is a model estimate, not a guaranteed outcome.";
  resultMessage.className = `message ${approved ? "success" : "warning"}`;
}

loanForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  updateDebtToIncomeRatio();

  const payload = {};
  for (const feature of FEATURE_IDS) {
    const input = document.getElementById(feature);
    if (input.value.trim() === "") {
      input.focus();
      showError(`Please complete "${feature.replaceAll("_", " ")}".`);
      return;
    }
    const value = Number(input.value);
    if (!Number.isFinite(value)) {
      input.focus();
      showError(`Please enter a valid number for ${feature.replaceAll("_", " ")}.`);
      return;
    }
    payload[feature] = value;
  }

  if (payload.annual_income <= 0) {
    annualIncomeInput.focus();
    showError("Annual income must be greater than zero to calculate the ratio.");
    return;
  }
  if (payload.existing_debt < 0 || payload.loan_amount < 0 ||
      payload.loan_term_months <= 0 || payload.credit_score < 0 ||
      payload.employment_years < 0 || payload.previous_defaults < 0) {
    showError("Please check your values. Amounts cannot be negative and loan term must be positive.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      throw new Error(data.error || "The server could not process this application.");
    }
    showResult(data);
  } catch (error) {
    showError(`${error.message} Check that app.py is running and model/scaler files are compatible.`);
  } finally {
    setLoading(false);
  }
});
