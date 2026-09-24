# Telco Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to leave (churn), so a retention team can contact them **before** they go.

The notebook covers the full workflow: data cleaning, exploratory analysis, preprocessing, model building and tuning, error analysis, model interpretation, and packaging the final model as a reusable scikit-learn pipeline.

## Problem statement

Winning a new customer costs much more than keeping an existing one. About **26.5%** of customers in this dataset churned, so the classes are imbalanced and plain accuracy is misleading (a model that always says "No churn" is already 73.5% accurate).

Because a missed churner costs more than an unnecessary retention offer, **recall on the churn class** is the main metric. Precision and F1 are tracked alongside it.

## Dataset

- **Telco Customer Churn** (IBM sample dataset, widely available on Kaggle)
- 7,043 customers, 21 columns: demographics, account information, subscribed services, monthly/total charges, and the `Churn` label
- Expected file name: `Telco-Customer-Churn.csv`, placed in the same folder as the notebook

## Approach

1. **Data cleaning**: `TotalCharges` was stored as text; blank values (11 customers with `tenure = 0`) were filled with 0 and the column converted to numeric.
2. **EDA**: target distribution, contract type, tenure groups, monthly charges.
3. **Preprocessing**: binary mapping and one-hot encoding, stratified 60/20/20 train/validation/test split, standard scaling of numeric columns.
4. **Modelling**: KNN baseline, then SMOTE for class imbalance, then hyperparameter tuning (grid search and a sweep over `k`), then comparison with Logistic Regression.
5. **Error analysis**: what the missed churners look like.
6. **Interpretation**: permutation importance.
7. **Final pipeline**: `ColumnTransformer` + SMOTE + KNN (`k=17`, distance-weighted) in one `imblearn` pipeline, saved with `joblib`.

## Results

Validation set (1,409 customers):

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Always predict "No churn" | 0.735 | 0.000 | - | - |
| KNN (k=5) | 0.758 | 0.508 | 0.548 | 0.527 |
| KNN (k=5) + SMOTE | 0.705 | 0.701 | 0.463 | 0.557 |
| KNN (k=17, distance-weighted) + SMOTE | 0.700 | **0.743** | 0.460 | 0.569 |
| Logistic Regression + SMOTE | 0.757 | 0.706 | 0.531 | 0.606 |

Final pipeline on the held-out **test set**: accuracy 0.711, precision 0.474, **recall 0.797**, F1 0.594.

### Key findings

- **Contract type is the strongest signal**: month-to-month customers churn at 42.7%, one-year at 11.3%, two-year at 2.8%.
- **The first year is the danger zone**: customers with 1-12 months of tenure churn at 47.7% and make up 55% of all churners.
- **Churners pay more**: about $74 per month on average versus $61 for retained customers.
- **SMOTE and tuning lift recall from 0.51 to about 0.74**, at the cost of precision (about 0.46).
- **Missed churners look like ordinary customers** (average tenure and charges), so they are hard to separate using the available features.

## Project structure

```
.
├── Customer_Churn_Prediction.ipynb   # main notebook
├── Telco-Customer-Churn.csv          # dataset (add it here)
├── requirements.txt
├── .gitignore
└── README.md
```

`churn_pipeline.pkl` is created when you run the notebook (it is git-ignored).

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Telco-Customer-Churn.csv to the project folder, then launch Jupyter
jupyter notebook Customer_Churn_Prediction.ipynb
```

Python 3.9 or newer is recommended.

## Using the saved model

After running the notebook, `churn_pipeline.pkl` contains the full pipeline (encoding, scaling, SMOTE, KNN). It accepts raw customer data:

```python
import joblib
import pandas as pd

pipeline = joblib.load("churn_pipeline.pkl")

customer = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 90.5, "TotalCharges": 452.5,
}

df = pd.DataFrame([customer])
print(pipeline.predict(df)[0])            # 1 = likely to churn
print(pipeline.predict_proba(df)[0][1])   # churn score
```

Pickle files depend on library versions, so load the model with the same scikit-learn version used to train it.

## Limitations

- KNN probabilities are neighbour vote shares: fine for ranking customers by risk, but not calibrated probabilities.
- The grid search ran on SMOTE-resampled data, so its cross-validation scores are optimistic.
- Precision is low (about 0.46-0.47), so many retention offers would go to customers who would have stayed.
- Results come from a single train/validation/test split.
- Only contract, tenure and monthly charges were explored in depth in the EDA.

## Next steps

- Try tree-based models (Random Forest, gradient boosting)
- Try class weights or a tuned decision threshold instead of SMOTE
- Choose the model and threshold using the actual cost of false negatives and false positives
- Calibrate predicted probabilities
- Wrap the pipeline in a small app or API

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib, Jupyter
