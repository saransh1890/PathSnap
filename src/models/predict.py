"""Run inference on test partition candidate sets using trained models."""
import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def predict_candidates(candidates_path, model_path, output_path, model_name="model"):
    """Score candidate arcs using a trained model and output ranked predictions."""
    print(f"Loading candidates from {candidates_path}...")
    df = pd.read_csv(candidates_path)
    
    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    print("Generating candidate match probabilities...")
    probs = pipeline.predict_proba(df)[:, 1]
    
    df_out = df.copy()
    score_col = f"{model_name}_match_prob"
    df_out[score_col] = probs
    
    # Compute relative ranking within each observation
    df_out[f"{model_name}_rank"] = (
        df_out.groupby(["trip_id", "observation_idx"], sort=False)[score_col]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    
    # Sort by trip, observation, and rank
    df_out = df_out.sort_values(["trip_id", "observation_idx", f"{model_name}_rank"])
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_file, index=False)
    print(f"Saved ranked predictions ({len(df_out)} rows) to {out_file}")
    
    # Print sample top predictions
    top1 = df_out[df_out[f"{model_name}_rank"] == 1]
    print(f"Total observations scored: {len(top1)}")
    print("Sample Top-1 candidate predictions:")
    print(top1[["trip_id", "observation_idx", "arc_id", "distance_m", score_col]].head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Generate candidate predictions for student test")
    parser.add_argument("--candidates", default="results/student_test_candidate_sample_k64.csv",
                        help="Path to student test candidates")
    parser.add_argument("--model", default="models/logistic_regression.joblib",
                        help="Path to trained model (.joblib)")
    parser.add_argument("--model-name", default="lr", help="Prefix name for score/rank columns")
    parser.add_argument("--out", default="results/student_test_predictions_lr.csv",
                        help="Output path for predictions")
    args = parser.parse_args()
    
    predict_candidates(args.candidates, args.model, args.out, model_name=args.model_name)


if __name__ == "__main__":
    main()
