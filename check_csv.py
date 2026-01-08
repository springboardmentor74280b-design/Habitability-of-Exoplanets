import pandas as pd
from web_app.services import process_csv, load_artifacts

def check_csv():
    load_artifacts()
    # Create a dummy CSV file mimicking valid input
    # We'll just read hwc.csv and take top 5 rows
    df = pd.read_csv("Dataset/hwc.csv").head(5)
    df.to_csv("temp_test.csv", index=False)
    
    print("Testing process_csv with temp_test.csv...")
    try:
        with open("temp_test.csv", "rb") as f:
            # Mocking FileStorage object? process_csv expects a file-like object or path?
            # services.py uses pd.read_csv(file_storage). pd.read_csv accepts file handle.
            result_df = process_csv(f)
            
        print("Result Columns:", result_df.columns.tolist())
        print("Result Head:\n", result_df.head())
        print("CSV Check: SUCCESS")
    except Exception as e:
        print(f"CSV Check: FAILED with error: {e}")

if __name__ == "__main__":
    check_csv()
