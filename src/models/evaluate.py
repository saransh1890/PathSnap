"""Comprehensive model evaluation and benchmarking for PathSnap."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate_ranking_and_classification(df_scored, score_col, method_name, ascending=False):
    """Compute standard classification and ranking metrics for candidate scoring.
    
    Args:
        df_scored: DataFrame containing candidates with 'label' and `score_col`.
        score_col: Name of column containing model prediction/score.
        method_name: String label for the evaluated method.
        ascending: True if lower score means better (e.g. distance_m), False if higher is better (probabilities).
    """
    df = df_scored.copy()
    y_true = df["label"].to_numpy(dtype=int)
    scores = df[score_col].to_numpy(dtype=float)
    
    # If scores are inverted (e.g. distance_m where lower is better), flip for AUC computation
    auc_scores = -scores if ascending else scores
    
    # Ranking metrics per observation
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    reciprocal_ranks = []
    
    grouped = df.groupby(["trip_id", "observation_idx"], sort=False)
    total_obs = len(grouped)
    
    trip_accuracies = {}
    
    for (trip_id, obs_idx), group in grouped:
        sorted_group = group.sort_values(score_col, ascending=ascending)
        labels_sorted = sorted_group["label"].to_numpy(dtype=int)
        
        # Check if positive candidate exists
        has_pos = (labels_sorted == 1).any()
        if not has_pos:
            continue
            
        pos_rank = np.where(labels_sorted == 1)[0][0] + 1  # 1-based rank
        reciprocal_ranks.append(1.0 / pos_rank)
        
        is_top1 = int(pos_rank == 1)
        is_top3 = int(pos_rank <= 3)
        is_top5 = int(pos_rank <= 5)
        
        top1_correct += is_top1
        top3_correct += is_top3
        top5_correct += is_top5
        
        if trip_id not in trip_accuracies:
            trip_accuracies[trip_id] = []
        trip_accuracies[trip_id].append(is_top1)
        
    top1_acc = top1_correct / total_obs if total_obs else 0.0
    top3_acc = top3_correct / total_obs if total_obs else 0.0
    top5_acc = top5_correct / total_obs if total_obs else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    
    # Top-1 binary prediction across all candidate rows
    # Assign prediction = 1 to the best candidate for each observation, 0 to others
    if ascending:
        best_scores = grouped[score_col].transform("min")
    else:
        best_scores = grouped[score_col].transform("max")
        
    y_pred = (df[score_col] == best_scores).astype(int).to_numpy().copy()
    
    # Handle ties: if multiple candidates have the same score, only keep the first per group
    # to maintain exactly 1 predicted positive per observation
    temp_df = pd.DataFrame({"y_pred": y_pred, "trip_id": df["trip_id"], "obs_idx": df["observation_idx"]})
    duplicated = temp_df.duplicated(subset=["trip_id", "obs_idx", "y_pred"]) & (temp_df["y_pred"] == 1)
    y_pred[duplicated.to_numpy()] = 0
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        roc_auc = float(roc_auc_score(y_true, auc_scores))
        pr_auc = float(average_precision_score(y_true, auc_scores))
    except Exception:
        roc_auc = np.nan
        pr_auc = np.nan
        
    trip_means = [np.mean(v) for v in trip_accuracies.values()]
    macro_top1 = float(np.mean(trip_means)) if trip_means else 0.0
    std_top1 = float(np.std(trip_means)) if trip_means else 0.0
    
    return {
        "method": method_name,
        "top_1_accuracy": round(top1_acc, 4),
        "top_3_accuracy": round(top3_acc, 4),
        "top_5_accuracy": round(top5_acc, 4),
        "mrr": round(mrr, 4),
        "macro_trip_top_1": round(macro_top1, 4),
        "std_trip_top_1": round(std_top1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "overall_accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PathSnap Models and Baseline")
    parser.add_argument("--candidates", default="results/train_candidate_sample_k64.csv",
                        help="Candidate CSV to evaluate on")
    parser.add_argument("--lr-model", default="models/logistic_regression.joblib",
                        help="Path to trained Logistic Regression pipeline")
    parser.add_argument("--knn-model", default="models/knn.joblib",
                        help="Path to trained KNN pipeline")
    parser.add_argument("--results-dir", default="results", help="Directory for output files")
    args = parser.parse_args()
    
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading candidate evaluation data from {args.candidates}...")
    df = pd.read_csv(args.candidates)
    df_eval = df[df["label"].isin([0, 1])].copy()
    print(f"Evaluation observations: {df_eval.groupby(['trip_id', 'observation_idx']).ngroups} observations across {df_eval['trip_id'].nunique()} trips")
    
    metrics_summary = []
    
    # 1. Geometric Nearest Baseline (classical naive baseline)
    print("\n[1/3] Evaluating Geometric Nearest Baseline (distance_m)...")
    res_base = evaluate_ranking_and_classification(
        df_eval, score_col="distance_m", method_name="Geometric Nearest Baseline", ascending=True
    )
    metrics_summary.append(res_base)
    
    # 2. Logistic Regression
    if Path(args.lr_model).exists():
        print(f"\n[2/3] Evaluating Logistic Regression ({args.lr_model})...")
        lr_pipeline = joblib.load(args.lr_model)
        df_eval["lr_prob"] = lr_pipeline.predict_proba(df_eval)[:, 1]
        res_lr = evaluate_ranking_and_classification(
            df_eval, score_col="lr_prob", method_name="Logistic Regression", ascending=False
        )
        metrics_summary.append(res_lr)
    else:
        print(f"Warning: {args.lr_model} not found, skipping.")
        
    # 3. K-Nearest Neighbours (KNN)
    if Path(args.knn_model).exists():
        print(f"\n[3/3] Evaluating K-Nearest Neighbours ({args.knn_model})...")
        knn_pipeline = joblib.load(args.knn_model)
        df_eval["knn_prob"] = knn_pipeline.predict_proba(df_eval)[:, 1]
        res_knn = evaluate_ranking_and_classification(
            df_eval, score_col="knn_prob", method_name="K-Nearest Neighbours (KNN)", ascending=False
        )
        metrics_summary.append(res_knn)
    else:
        print(f"Warning: {args.knn_model} not found, skipping.")
        
    df_summary = pd.DataFrame(metrics_summary)
    
    # Save artifacts
    csv_path = res_dir / "model_evaluation_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved evaluation summary table to {csv_path}")
    
    json_path = res_dir / "model_comparison_report.json"
    json_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"Saved evaluation JSON report to {json_path}")
    
    # Display formatted comparison table
    print("\n================================ MODEL EVALUATION BENCHMARK ================================")
    cols_display = ["method", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "mrr", "macro_trip_top_1", "roc_auc", "f1_score"]
    print(df_summary[cols_display].to_string(index=False))
    print("============================================================================================\n")


if __name__ == "__main__":
    main()
