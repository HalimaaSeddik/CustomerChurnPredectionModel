
import matplotlib.pyplot as plt
import pandas as pd 
# Keep the original data unchanged
file_path ="data/raw/Telco-Customer-Churn.csv"
df =pd.read_csv(file_path)
df_clean = df.copy()

# Convert blank strings into missing values
df_clean["TotalCharges"] = (
    df_clean["TotalCharges"]
    .replace(r"^\s*$", pd.NA, regex=True)
)

# Convert the column to numeric
df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"],
    errors="coerce"
)

# Check whether missing values correspond to zero tenure
mask = (
    df_clean["TotalCharges"].isna()
    & (df_clean["tenure"] == 0)
)

print("Missing TotalCharges with zero tenure:", mask.sum())

# Fill these specific missing values with zero
df_clean.loc[mask, "TotalCharges"] = 0

# Verify the result
print("\nTotalCharges data type:")
print(df_clean["TotalCharges"].dtype)

print("\nRemaining missing values:")
print(df_clean["TotalCharges"].isna().sum())

print("\nRows with zero tenure:")
print(
    df_clean.loc[
        df_clean["tenure"] == 0,
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ]
)





# Numerical features
numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Summary statistics
print("Numerical summary:")
print(df_clean[numerical_cols].describe().T)

# Histograms
df_clean[numerical_cols].hist(
    bins=30,
    figsize=(12, 7),
    edgecolor="black"
)

plt.suptitle("Distributions of Numerical Features")
plt.tight_layout()
plt.show()

# Boxplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, col in zip(axes, numerical_cols):
    ax.boxplot(df_clean[col].dropna())
    ax.set_title(col)
    ax.set_ylabel("Value")

plt.tight_layout()
plt.show()





# Calculate churn rate for each contract type
churn_by_contract = pd.crosstab(
    df_clean["Contract"],
    df_clean["Churn"],
    normalize="index"
) * 100

print(churn_by_contract.round(2))

# Plot churn rate
churn_rate = churn_by_contract["Yes"].sort_values()

churn_rate.plot(kind="bar", figsize=(8, 5))
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()






# Compare tenure for customers who churned and those who stayed
df_clean.boxplot(
    column="tenure",
    by="Churn",
    figsize=(8, 5)
)

plt.title("Customer Tenure by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.show()

# Compare average and median tenure
print(
    df_clean.groupby("Churn")["tenure"]
    .agg(["mean", "median", "count"])
    .round(2)
)


# Calculate churn percentage for each internet service
churn_by_internet = pd.crosstab(
    df_clean["InternetService"],
    df_clean["Churn"],
    normalize="index"
) * 100

print("Churn rate by Internet Service:")
print(churn_by_internet.round(2))

# Plot churn rate
churn_by_internet["Yes"].sort_values().plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Compare monthly charges by internet service
charges_by_internet = (
    df_clean.groupby("InternetService")["MonthlyCharges"]
    .agg(["mean", "median", "count"])
    .round(2)
)

print(charges_by_internet)

# Compare monthly charges for churned and retained customers
print("\nMonthly charges by Internet Service and Churn:")

print(
    df_clean.groupby(["InternetService", "Churn"])["MonthlyCharges"]
    .agg(["mean", "median", "count"])
    .round(2)
)



# Create four groups based on monthly charges
df_clean["ChargeGroup"] = pd.qcut(
    df_clean["MonthlyCharges"],
    q=4,
    duplicates="drop"
)

# Calculate churn rate within each charge group
churn_by_charges = (
    df_clean.groupby("ChargeGroup", observed=True)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print(churn_by_charges.round(2))


churn_by_contract_and_charges = (
    df_clean.groupby(
        ["Contract", "ChargeGroup"],
        observed=True
    )["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .unstack()
)

print(churn_by_contract_and_charges.round(2))



numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

correlation_matrix = df_clean[numerical_cols].corr()

print(correlation_matrix.round(2))

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")
plt.xticks(range(3), numerical_cols, rotation=30)
plt.yticks(range(3), numerical_cols)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# 1. Exact correlation coefficients
numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

print("Correlation matrix:")
print(df_clean[numerical_cols].corr().round(3))


# 2. Churn rate for every categorical feature
categorical_cols = [
    col for col in df_clean.select_dtypes(include=["object", "str"]).columns
    if col not in ["customerID", "Churn"]
]

print("\nChurn rates by categorical feature:")

for col in categorical_cols:
    rates = (
        df_clean.groupby(col, dropna=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    print(f"\n--- {col} ---")
    print(rates.round(2))