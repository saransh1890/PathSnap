"""Train and benchmark 5 distinct ML model families for candidate arc ranking."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, f1_score
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_engineering import CandidateFeaturePipeline


def compute_observation_metrics(df_eval, prob_col="pred_prob"):
    """Compute Top-1, Top-3, Top-5 accuracy and MRR per observation."""
    grouped = df_eval.groupby(["trip_id", "observation_idx"], sort=False)
    total_obs = len(grouped)
    if total_obs == 0:
        return {"top_1_acc": 0.0, "top_3_acc": 0.0, "top_5_acc": 0.0, "mrr": 0.0}
        
    top1 = 0
    top3 = 0
    top5 = 0
    mrr_list = []
    
    for _, group in grouped:
        sorted_g = group.sort_values(prob_col, ascending=False)
        labels = sorted_g["label"].to_numpy(dtype=int)
        if (labels == 1).any():
            rank = np.where(labels == 1)[0][0] + 1
            mrr_list.append(1.0 / rank)
            top1 += int(rank == 1)
            top3 += int(rank <= 3)
            top5 += int(rank <= 5)
            
    return {
        "top_1_acc": float(top1 / total_obs),
        "top_3_acc": float(top3 / total_obs),
        "top_5_acc": float(top5 / total_obs),
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0
    }


def train_and_benchmark(candidates_path, models_dir="models", results_dir="results"):
    models_path = Path(models_dir)
    res_path = Path(results_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading candidate dataset from {candidates_path}...")
    df = pd.read_csv(candidates_path)
    df_clean = df[df["label"].isin([0, 1])].copy()
    
    print(f"Total rows: {len(df_clean)}, Unique trips: {df_clean['trip_id'].nunique()}")
    
    # Stratified/hard-negative sample for fast scalable training
    positives = df_clean[df_clean["label"] == 1]
    negatives = df_clean[df_clean["label"] == 0]
    closest_neg = (
        negatives.sort_values(["trip_id", "observation_idx", "distance_m"])
        .groupby(["trip_id", "observation_idx"], sort=False)
        .head(6)
    )
    df_train = pd.concat([positives, closest_neg], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Balanced training subset: {len(df_train)} rows (Pos={sum(df_train['label']==1)}, Neg={sum(df_train['label']==0)})")
    
    y_train = df_train["label"].to_numpy(dtype=int)
    groups = df_train["trip_id"].to_numpy()
    
    # Define the 5 distinct model families
    model_definitions = {
        "Logistic Regression": {
            "family": "Linear / Probabilistic",
            "model": LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42),
            "file": "logistic_regression.joblib"
        },
        "K-Nearest Neighbours": {
            "family": "Instance-based",
            "model": KNeighborsClassifier(n_neighbors=21, weights="distance", n_jobs=-1),
            "file": "knn.joblib"
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
            "file": "random_forest.joblib"
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "model": HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.08, random_state=42),
            "file": "hist_gradient_boosting.joblib"
        },
        "Multi-Layer Perceptron (MLP)": {
            "family": "Neural Network",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, alpha=0.01, random_state=42),
            "file": "mlp_classifier.joblib"
        }
    }
    
    benchmark_results = []
    
    for name, spec in model_definitions.items():
        print(f"\nEvaluating Model: {name} ({spec['family']})...")
        pipeline = Pipeline([
            ("features", CandidateFeaturePipeline()),
            ("clf", spec["model"])
        ])
        
        # 5-Fold GroupKFold Cross-Validation on trips
        gkf = GroupKFold(n_splits=5)
        cv_aucs, cv_top1s, cv_f1s = [], [], []
        
        t0 = time.time()
        for train_idx, val_idx in gkf.split(df_train, y_train, groups):
            train_fold = df_train.iloc[train_idx]
            val_fold = df_train.iloc[val_idx].copy()
            
            pipeline.fit(train_fold, train_fold["label"])
            val_probs = pipeline.predict_proba(val_fold)[:, 1]
            val_fold["pred_prob"] = val_probs
            
            cv_aucs.append(roc_auc_score(val_fold["label"], val_probs))
            top_res = compute_observation_metrics(val_fold, "pred_prob")
            cv_top1s.append(top_res["top_1_acc"])
            val_pred = (val_probs >= 0.5).astype(int)
            cv_f1s.append(f1_score(val_fold["label"], val_pred, zero_division=0))
            
        fit_time = round(time.time() - t0, 2)
        
        # Fit on full dataset and save artifact
        t_fit_start = time.time()
        pipeline.fit(df_train, y_train)
        train_time = round(time.time() - t_fit_start, 3)
        
        model_save_path = models_path / spec["file"]
        joblib.dump(pipeline, model_save_path)
        print(f"  -> Saved artifact to {model_save_path}")
        
        # Full evaluation inference
        t_inf_start = time.time()
        eval_probs = pipeline.predict_proba(df_clean)[:, 1]
        inference_time_ms = round((time.time() - t_inf_start) / len(df_clean) * 1000, 3)
        
        df_clean["pred_prob"] = eval_probs
        obs_metrics = compute_observation_metrics(df_clean, "pred_prob")
        
        eval_pred = (eval_probs >= 0.5).astype(int)
        y_true = df_clean["label"].to_numpy(dtype=int)
        
        row = {
            "model": name,
            "algorithm_family": spec["family"],
            "cv_macro_f1": round(float(np.mean(cv_f1s)), 4),
            "top_1_accuracy": round(obs_metrics["top_1_acc"], 4),
            "top_3_accuracy": round(obs_metrics["top_3_acc"], 4),
            "top_5_accuracy": round(obs_metrics["top_5_acc"], 4),
            "mrr": round(obs_metrics["mrr"], 4),
            "roc_auc": round(float(roc_auc_score(y_true, eval_probs)), 4),
            "pr_auc": round(float(average_precision_score(y_true, eval_probs)), 4),
            "log_loss": round(float(log_loss(y_true, eval_probs)), 4),
            "training_time_s": train_time,
            "inference_time_ms": inference_time_ms
        }
        benchmark_results.append(row)
        print(f"  -> Top-1 Acc: {row['top_1_accuracy']*100:.2f}% | Top-3 Acc: {row['top_3_accuracy']*100:.2f}% | ROC-AUC: {row['roc_auc']:.4f}")
        
    df_bench = pd.DataFrame(benchmark_results)
    csv_out = res_path / "all_models_benchmark_summary.csv"
    df_bench.to_csv(csv_out, index=False)
    print(f"\nSaved 5-Model Benchmark Summary to {csv_out}")
    
    json_out = res_path / "all_models_benchmark.json"
    json_out.write_text(json.dumps(benchmark_results, indent=2))
    
    print("\n" + "=" * 90)
    print("🏆 FINAL CONSOLIDATED 5-MODEL BENCHMARK TABLE")
    print("=" * 90)
    print(df_bench[["model", "algorithm_family", "top_1_accuracy", "top_3_accuracy", "cv_macro_f1", "roc_auc", "log_loss", "inference_time_ms"]].to_string(index=False))
    print("=" * 90 + "\n")
    return df_bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="results/train_candidate_sample_k64.csv")
    args = parser.parse_args()
    train_and_benchmark(args.candidates)


if __name__ == "__main__":
    main()
