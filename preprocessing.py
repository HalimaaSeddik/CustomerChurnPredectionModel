from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from data_preprocessing import load_and_clean_data
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# load cleaned data
df_clean =load_and_clean_data("data/raw/Telco-Customer-Churn.csv")



# Features: information used to predict churn
X = df_clean.drop(columns=["Churn", "customerID"])

# Target: what we want to predict
y = df_clean["Churn"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True).mul(100).round(2))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).mul(100).round(2))



# Identify numerical and categorical features

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"] 
).columns.tolist()
numerical_features.remove("SeniorCitizen")  # Remove SeniorCitizen from numerical features

categorical_features = X_train.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist() 
categorical_features.append("SeniorCitizen")  # Add SeniorCitizen to categorical features
print("Numerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nTotal features detected:",
      len(numerical_features) + len(categorical_features))


# Numerical features: scale their values
numerical_transformer = StandardScaler()

# Categorical features: convert categories to numeric columns
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# Apply the appropriate transformer to each column group
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessor created successfully!")

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

print(model)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model trained successfully!")
print("Predictions:", y_pred[:10])



# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 4))

# 2. Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 3. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))