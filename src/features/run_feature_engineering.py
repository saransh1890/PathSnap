"""CLI runner to extract and export engineered feature matrices."""
import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_engineering import engineer_candidate_features, FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Engineer candidate features for PathSnap")
    parser.add_argument("--candidates", default="results/train_candidate_sample_k64.csv",
                        help="Input candidate CSV")
    parser.add_argument("--out", default="results/train_engineered_features_sample.csv",
                        help="Output path for engineered features CSV")
    parser.add_argument("--rows", type=int, default=None, help="Optional row limit for quick inspection")
    args = parser.parse_args()
    
    print(f"Loading candidates from {args.candidates}...")
    df = pd.read_csv(args.candidates, nrows=args.rows)
    print(f"Loaded {len(df)} candidate rows.")
    
    print("Engineering 19 spatial, angular, kinematic, and ranking features...")
    df_feat = engineer_candidate_features(df)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_path, index=False)
    print(f"Saved engineered feature table ({len(df_feat)} rows, {len(df_feat.columns)} columns) to {out_path}")
    
    print("\nEngineered Feature Columns:")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i:>2}. {col:<30} (dtype: {df_feat[col].dtype})")


if __name__ == "__main__":
    main()
