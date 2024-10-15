import pandas as pd


def convert_parquet_to_csv(parquet_file, csv_file):
    try:
        df = pd.read_parquet(parquet_file)
        df.to_csv(csv_file, index=False)
        print(f"Conversion successful! '{csv_file}' has been created.")
    except Exception as e:
        print(f"An error occurred during the conversion: {e}")


parquet_file = "data/open_llm_leaderboard.parquet"
csv_file = "data/open_llm_leaderboard.csv"

convert_parquet_to_csv(parquet_file, csv_file)
