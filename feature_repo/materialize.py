from feast import FeatureStore
import pandas as pd
import os

def main():
    fs = FeatureStore(repo_path='.')
    processed = os.path.join("..","data","processed","stock_data.parquet")
    df = pd.read_parquet(processed)
    start = df['timestamp'].min()
    end = df['timestamp'].max()
    print(f"Materializing from {start} to {end}")
    fs.materialize(start_date=start, end_date=end)
    print("✅ Materialization complete")

if __name__ == '__main__':
    main()
