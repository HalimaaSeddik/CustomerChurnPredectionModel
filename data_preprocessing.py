import pandas as pd

def load_and_clean_data(file_path):
    # Load the original dataset
    df = pd.read_csv(file_path)

    # Work on a copy
    df_clean = df.copy()

    # Convert blank strings to missing values
    df_clean["TotalCharges"] = (
        df_clean["TotalCharges"]
        .replace(r"^\s*$", pd.NA, regex=True)
    )

    # Convert TotalCharges to numeric
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"],
        errors="coerce"
    )

    # For customers with zero tenure and missing charges,
    # set TotalCharges to zero
    mask = (
        df_clean["TotalCharges"].isna()
        & (df_clean["tenure"] == 0)
    )

    df_clean.loc[mask, "TotalCharges"] = 0

    return df_clean


if __name__ == "__main__":
    df_clean = load_and_clean_data(
        "data/raw/Telco-Customer-Churn.csv"
    )

    print(df_clean.dtypes)
    print("Missing TotalCharges:",
          df_clean["TotalCharges"].isna().sum())