from datasets import load_dataset
import pandas as pd


def load_data():
    """Load and preprocess the data from the open-llm-leaderboard dataset.

    Returns:
        None: Saves the data directly to a CSV file with date timestamp
    """
    try:
        from datetime import datetime

        # Get current date in YYYY_MM_DD format
        date_str = datetime.now().strftime("%Y_%m_%d")
        filename = f"data/open_llm_leaderboard_{date_str}.csv"

        dataset = load_dataset("open-llm-leaderboard/contents")
        df = pd.DataFrame(dataset["train"])
        # Save to CSV with timestamp
        df.to_csv(filename, index=False)
        print(f"Data successfully downloaded and saved to {filename}")
    except Exception as e:
        print(f"Error loading data: {e}")


if __name__ == "__main__":
    load_data()
