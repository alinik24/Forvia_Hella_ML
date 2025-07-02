# data_cleaning_bookings.py

import pandas as pd


# TODO Adjust the file path as needed
DATA_FILE = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data\bookings_encoded_data.parquet"


def analyze_missingness(df):
    print("\n=== Missingness Indicators Analysis ===")
    nan_cols = [col for col in df.columns if col.endswith('_nan')]
    print(f"Found {len(nan_cols)} missingness indicator columns.")

    total_rows = df.shape[0]

    print("\nMissingness count and percentage per '_nan' column:")
    nan_summary = {}
    for col in nan_cols:
        missing_count = df[col].sum()  # sum of 1s = missing occurrences
        missing_pct = 100 * missing_count / total_rows
        nan_summary[col] = missing_pct
        print(f"{col}: {missing_count} ({missing_pct:.2f}%)")

    # Row-level missingness: how many missing flags per row
    df['missing_indicator_count'] = df[nan_cols].sum(axis=1)
    print("\nRow-level missingness indicator counts:")
    print(df['missing_indicator_count'].describe())

    no_missing = (df['missing_indicator_count'] == 0).sum()
    some_missing = (df['missing_indicator_count'] > 0).sum()
    print(f"\nRows with no missing indicators: {no_missing} ({100 * no_missing / total_rows:.2f}%)")
    print(f"Rows with at least one missing indicator: {some_missing} ({100 * some_missing / total_rows:.2f}%)")

    threshold = 5
    many_missing = df[df['missing_indicator_count'] > threshold]
    print(f"\nRows with more than {threshold} missing indicators: {many_missing.shape[0]}")

    # Uncomment to drop rows with too many missing indicators:
    # df_cleaned = df[df['missing_indicator_count'] <= threshold].copy()
    # print(f"Dropped {many_missing.shape[0]} rows with >{threshold} missing indicators.")
    # return df_cleaned

    return df

def main():
    print(f"Loading dataset from {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    print(f"Dataset shape: {df.shape}")

    print("\nGeneral dataframe info:")
    print(df.info())

    print("\nBasic stats summary:")
    print(df.describe())

    # Analyze missingness indicators
    df = analyze_missingness(df)

    # Save summary or cleaned dataset if needed
    # df.to_parquet("filtered_bookings_cleaned.parquet", index=False)
    print("\nData cleaning analysis complete.")

if __name__ == '__main__':
    main()