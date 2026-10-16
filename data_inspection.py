import pandas as pd 
file_path ="data/raw/Telco-Customer-Churn.csv"
df =pd.read_csv(file_path)

print("first5 rows\n",df.head())
print("dataset shape\n",df.shape)
print("column names\n",df.columns.tolist())
print("missing values",df.isnull().sum())
print("data info ",df.info())
print("statistical summary ",df.describe(include="all").T)
print("check the actual data types ",df.dtypes)
print("totalcharges",df["TotalCharges"].head(15).to_list())
print("count empty strings in total charges clmn")

print((df["TotalCharges"].astype(str).str.strip()=="").sum())
print("check duplicate",df.duplicated().sum())
print(df[df["tenure"]==0][["customerID","tenure","TotalCharges","MonthlyCharges"]])
print("check missing values ")
for col in df.columns:
    blank_count =(
        df[col].astype("string").str.strip().eq("").sum()
    )
    missing_count=df[col].isnull().sum()
    if missing_count >0 or blank_count>0:
        print(f"{col}: missing values:{missing_count},"
        f"blanck_values:{blank_count}")

print(df["Churn"].value_counts())
print((df["Churn"].value_counts(normalize=True)*100).round(2))




