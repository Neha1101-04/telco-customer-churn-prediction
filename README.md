# Customer Churn Prediction: Telecom

Machine learning project that identifies which telecom customers are likely to leave (churn), so a retention team can contact them **before** they go.

**Final model:** Logistic Regression (`class_weight="balanced"`, `C=3`) inside a leak-free scikit-learn pipeline.
**Test-set result (1,409 unseen customers):** 78% churn recall, 50% churn precision, ROC-AUC 0.841, PR-AUC 0.629.

## Business problem

Winning a new customer costs far more than keeping an existing one. A model that flags at-risk customers lets the retention team focus offers where they matter.

Only about 26.5% of customers churn, so the classes are imbalanced. A model that predicts "nobody churns" already scores 73.5% accuracy, so accuracy is **not** used as the headline metric. Models are judged on recall, precision, F1, ROC-AUC and PR-AUC for the churn class.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers and 21 columns (demographics, services, contract, billing and the churn label). Each row is one customer and the target is `Churn` (Yes / No).

The CSV is not included in this repository. Download it and save it as `data/Telco-Customer-Churn.csv`.

## Approach

1. **Data cleaning:** `TotalCharges` is stored as text. Its 11 blank values all belong to customers with `tenure = 0` (not billed yet), so they are set to 0 rather than dropped or guessed.
2. **EDA:** class balance, churn rate by segment, tenure, monthly charges and numeric correlations.
3. **Leak-free preprocessing:** the data is split first (80/20, stratified). Scaling, one-hot encoding and SMOTE live inside a `Pipeline`, so they are refit on each cross-validation fold.
4. **Model comparison:** Dummy baseline, KNN + SMOTE, Logistic Regression, Random Forest and Gradient Boosting, compared with 5-fold stratified CV on the training set.
5. **Tuning:** `GridSearchCV` on the full pipeline for KNN and Logistic Regression, optimising F1 on the churn class.
6. **Final evaluation:** the test set is scored once.
7. **Interpretation:** coefficients, permutation importance, error analysis and a decision-threshold trade-off.
8. **Deployment-ready artifact:** the whole pipeline is saved with `joblib` and accepts raw customer data.

## Results

| Model (5-fold CV, training set) | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Gradient Boosting | 0.799 | 0.632 | 0.848 | 0.667 |
| Logistic Regression | 0.803 | 0.629 | 0.846 | 0.660 |
| Random Forest | 0.730 | 0.633 | 0.844 | 0.656 |
| KNN + SMOTE | 0.775 | 0.585 | 0.796 | 0.549 |
| Dummy baseline | 0.000 | 0.000 | 0.500 | 0.265 |

Logistic Regression, Random Forest and Gradient Boosting are statistically indistinguishable, so the simplest and most interpretable one was chosen.

| Test set (threshold 0.5) | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| KNN + SMOTE (first model) | 0.459 | 0.874 | 0.602 | 0.827 | 0.608 |
| **Logistic Regression (final)** | 0.505 | 0.783 | 0.614 | 0.841 | 0.629 |

- The final model catches 293 of 374 real churners.
- Contacting the top 30% of customers by risk score reaches about 64% of all churners, which is 2.2x better than random.
- Test scores are close to cross-validated scores (F1 0.61 vs 0.63), which suggests no overfitting or leakage.

## Key findings

- **Contract type, tenure and internet service are the strongest churn signals.** Month-to-month customers churn at about 43% (two-year contracts: about 3%). Customers in their first year churn at about 47%.
- Fiber-optic customers (about 42%), electronic-check payers (about 45%) and customers without Tech Support or Online Security churn far more than average.
- The model catches the textbook churner (new, month-to-month, fiber) but misses churners who look loyal. Better performance would need new data such as support tickets, usage or competitor pricing.

## Limitations

- About half of the flagged customers are false alarms (precision about 50%).
- The output is a **risk score for ranking customers**, not a calibrated probability.
- Static snapshot: no time, usage, complaint or competitor data.
- Coefficients show association, not causation, and several features are correlated (`tenure`, `TotalCharges`, `MonthlyCharges`).

## Getting started

```bash
git clone https://github.com/Neha1101-04/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place the dataset at data/Telco-Customer-Churn.csv, then:
jupyter notebook
```

Open the notebook and run all cells. The trained pipeline is written to `models/churn_pipeline.joblib`.

## Predicting for a new customer

The saved object is the entire pipeline (preprocessing + model), so it takes raw feature values exactly as they appear in the source table.

```python
import joblib
import pandas as pd

model = joblib.load("models/churn_pipeline.joblib")

customer = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.5, "TotalCharges": 452.5,
}

score = model.predict_proba(pd.DataFrame([customer]))[0, 1]
print("Churn" if score >= 0.5 else "Stay", round(score, 3))
```

## Repository structure

```
.
├── Customer_Churn_Prediction.ipynb   # full analysis (rename to match your file)
├── data/
│   └── Telco-Customer-Churn.csv      # not tracked, download separately
├── models/
│   └── churn_pipeline.joblib         # created by the notebook
├── requirements.txt
└── README.md
```

## Next steps

- Choose the decision threshold from real costs (offer cost vs customer lifetime value).
- Calibrate probabilities (`CalibratedClassifierCV`) and add SHAP explanations.
- Feature engineering (number of services, charge per month of tenure) and tuned XGBoost / LightGBM.
- Wrap the pipeline in an API or Streamlit app and add drift monitoring.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn, joblib.

## Author

**Neha Edwin**: B.Tech Computer Science & Engineering student
[LinkedIn](https://www.linkedin.com/in/neha-edwin-5316b2298) · [GitHub](https://github.com/Neha1101-04) · neha1101@gmail.com
