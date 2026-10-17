# Customer Churn Prediction

A machine learning project to predict whether a telecom customer is likely to leave (Churn) using account, service, and billing information.

Status: Baseline Logistic Regression trained and evaluated. This is a learning project and has not been validated for production use.

## Goals
- Clean and prepare the dataset.
- Explore churn patterns with EDA.
- Build a reproducible preprocessing/model pipeline.
- Evaluate a baseline classifier and compare other models.

## Dataset
The Telco Customer Churn dataset contains 7,043 records and 21 original columns. Churn is the target (Yes / No). customerID is excluded because it is an identifier, not a useful predictive feature.

Check the dataset's redistribution terms before uploading the CSV to a public repository. If permission is unclear, do not commit the dataset; provide download instructions instead.

## Workflow
1. Clean blank TotalCharges values and convert the column to numeric.
2. Replace the 11 missing TotalCharges values associated with tenure == 0 with zero.
3. Explore distributions and churn rates with EDA.
4. Split data 80/20 with random_state=42 and stratify=y.
5. Use StandardScaler for numeric features and OneHotEncoder(handle_unknown="ignore") for categorical features.
6. Train a LogisticRegression(max_iter=1000) baseline in a scikit-learn Pipeline.
7. Evaluate with Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

## Baseline results

| Metric | No | Yes |
|---|---:|---:|
| Precision | 0.85 | 0.66 |
| Recall | 0.89 | 0.56 |
| F1-score | 0.87 | 0.60 |
| Support | 1035 | 374 |

- Accuracy: 0.8055 (80.55%)
- Confusion matrix: [[926, 109], [165, 209]], class order No, Yes.

The model correctly classified 926 non-churn customers and 209 churn customers. It produced 109 false positives and 165 false negatives. Recall for Yes is approximately 56%, so the baseline misses a substantial proportion of customers who churned. Accuracy alone is insufficient because the classes are imbalanced.

## Suggested structure

CustomerChurn/
â”œâ”€â”€ data/
â”‚   â””â”€â”€ raw/
â”‚       â””â”€â”€ Telco-Customer-Churn.csv
â”œâ”€â”€ data_processing.py
â”œâ”€â”€ preprocessing.py
â”œâ”€â”€ eda.py
â”œâ”€â”€ app.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
Adjust paths and filenames to match your repository.

## Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
On Windows, activate with .venv\\Scripts\\activate.

## Run

python data_processing.py
python preprocessing.py
streamlit run app.py
The Streamlit demo expects the CSV at data/raw/Telco-Customer-Churn.csv; update the path in app.py if needed.

## Metrics
- Accuracy: proportion of all predictions that are correct.
- Precision: among predictions for a class, proportion that truly belong to it.
- Recall: among actual examples of a class, proportion identified by the model.
- F1-score: harmonic mean of precision and recall.
- Support: number of true examples in each class.
- Confusion matrix: counts true negatives, false positives, false negatives, and true positives.
- Data leakage: using test information during training or preprocessing.

## Next steps
- Compare Logistic Regression with Decision Tree and Random Forest.
- Use the same split and metrics for fair comparison.
- Consider cross-validation and hyperparameter tuning.
- Save the final pipeline and document limitations before deployment.

## Limitations
- Results come from one train/test split.
- The baseline has not been tuned.
- Predictions are not guaranteed to generalize to current customers.
- No real-world deployment or business validation has been performed.

## License and attribution
Add the dataset source and its license/usage terms here. Choose a license for your own code separately; do not imply the dataset is covered by your code license unless it is.
