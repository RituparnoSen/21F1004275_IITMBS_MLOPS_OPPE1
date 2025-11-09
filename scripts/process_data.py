#!/usr/bin/env python3
import os, glob, json
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIRS = ["data/raw/v0", "data/raw/v1"]
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def infer_stock_name_from_filename(fn):
    return Path(fn).stem.split("__")[0]

def load_all_csvs():
    dfs=[]
    for d in RAW_DIRS:
        for f in sorted(glob.glob(os.path.join(d,"*.csv"))):
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            df['stock_name'] = infer_stock_name_from_filename(f)
            dfs.append(df)
            print(f"Loaded {f} rows={len(df)}")
    if not dfs:
        raise SystemExit("No CSVs found.")
    return pd.concat(dfs, ignore_index=True)

def prepare_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['stock_name','timestamp']).reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    def compute(g):
        g = g.reset_index(drop=True)
        g['rolling_avg_10'] = g['close'].rolling(window=10, min_periods=1).mean()
        g['volume_sum_10'] = g['volume'].rolling(window=10, min_periods=1).sum()
        g['close_t_plus_5'] = g['close'].shift(-5)
        g['target_5min'] = (g['close_t_plus_5'] > g['close']).astype('Int64')
        return g

    df = df.groupby('stock_name', group_keys=False).apply(compute).reset_index(drop=True)
    df = df.dropna(subset=['target_5min'])
    return df

def split_train_test(df, test_frac=0.2):
    train_parts=[]
    test_parts=[]
    splits = {}
    for stock, g in df.groupby('stock_name'):
        g = g.sort_values('timestamp')
        n = len(g)
        test_n = int(round(test_frac*n))
        train = g.iloc[:(n-test_n)]
        test = g.iloc[(n-test_n):]
        train_parts.append(train)
        test_parts.append(test)
        splits[stock] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": str(train['timestamp'].min()),
            "train_end": str(train['timestamp'].max()),
            "test_start": str(test['timestamp'].min()),
            "test_end": str(test['timestamp'].max())
        }
    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)
    return train_df, test_df, splits

def main():
    print("Loading raw data...")
    raw = load_all_csvs()
    print("Processing features...")
    processed = prepare_features(raw)
    print("Splitting data...")
    train_df, test_df, splits = split_train_test(processed, test_frac=float(os.getenv("TEST_SIZE",0.2)))

    processed.to_parquet(OUT_DIR/"stock_data.parquet", index=False)
    train_df.to_parquet(OUT_DIR/"train.parquet", index=False)
    test_df.to_parquet(OUT_DIR/"test.parquet", index=False)
    with open(OUT_DIR/"splits.json","w") as f: json.dump(splits, f, indent=2)

    print(f"Processed={len(processed)}, Train={len(train_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    main()
