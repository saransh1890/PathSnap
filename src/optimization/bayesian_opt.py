"""Bayesian Hyperparameter Optimization for PathSnap using Gaussian Process surrogate."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_engineering import CandidateFeaturePipeline


def compute_top1_acc(df_eval, prob_col="pred_prob"):
    grouped = df_eval.groupby(["trip_id", "observation_idx"], sort=False)
    total = len(grouped)
    if total == 0:
        return 0.0
    correct = 0
    for _, group in grouped:
        top_cand = group.nlargest(1, prob_col)
        if (top_cand["label"] == 1).any():
            correct += 1
    return float(correct / total)


def run_bayesian_optimization(candidates_path, n_calls=12, results_dir="results", models_dir="models"):
    res_path = Path(results_dir)
    mod_path = Path(models_dir)
    res_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {candidates_path} for Bayesian Hyperparameter Tuning...")
    df = pd.read_csv(candidates_path)
    df_clean = df[df["label"].isin([0, 1])].copy()
    
    # Subsample hard negatives for fast cross-validation
    positives = df_clean[df_clean["label"] == 1]
    negatives = df_clean[df_clean["label"] == 0]
    closest_neg = (
        negatives.sort_values(["trip_id", "observation_idx", "distance_m"])
        .groupby(["trip_id", "observation_idx"], sort=False)
        .head(6)
    )
    df_train = pd.concat([positives, closest_neg], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Pre-extract features once to accelerate optimization trials
    feat_pipe = CandidateFeaturePipeline()
    X_train = feat_pipe.fit_transform(df_train)
    y_train = df_train["label"].to_numpy(dtype=int)
    groups = df_train["trip_id"].to_numpy()
    
    # Feature transform full dataset for final evaluation
    X_full = feat_pipe.transform(df_clean)
    y_full = df_clean["label"].to_numpy(dtype=int)
    
    # 1. Evaluate baseline untuned model
    print("\n[1/3] Evaluating Untuned Default Model (HistGradientBoosting)...")
    base_clf = HistGradientBoostingClassifier(random_state=42)
    t0 = time.time()
    base_clf.fit(X_train, y_train)
    t_inf0 = time.time()
    base_probs = base_clf.predict_proba(X_full)[:, 1]
    untuned_inf_time = round((time.time() - t_inf0) / len(df_clean) * 1000, 4)
    df_clean["pred_prob"] = base_probs
    untuned_top1 = compute_top1_acc(df_clean, "pred_prob")
    untuned_loss = float(log_loss(y_full, base_probs))
    untuned_f1 = float(f1_score(y_full, (base_probs >= 0.5).astype(int), zero_division=0))
    untuned_auc = float(roc_auc_score(y_full, base_probs))
    
    print(f"  Untuned Top-1 Acc: {untuned_top1*100:.2f}% | F1: {untuned_f1:.4f} | Log Loss: {untuned_loss:.4f}")
    
    # 2. Define Bayesian Search Space
    print(f"\n[2/3] Setting up Gaussian Process Bayesian Optimization ({n_calls} trials)...")
    space = [
        Real(0.02, 0.20, name="learning_rate", prior="log-uniform"),
        Integer(50, 150, name="max_iter"),
        Integer(4, 12, name="max_depth"),
        Integer(10, 40, name="min_samples_leaf"),
        Real(1e-4, 5.0, name="l2_regularization", prior="log-uniform")
    ]
    
    trials_history = []
    gkf = GroupKFold(n_splits=3)
    
    @use_named_args(space)
    def objective(**params):
        clf = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=42
        )
        fold_scores = []
        for tr_idx, val_idx in gkf.split(X_train, y_train, groups):
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            probs = clf.predict_proba(X_train[val_idx])[:, 1]
            fold_scores.append(roc_auc_score(y_train[val_idx], probs))
            
        mean_score = float(np.mean(fold_scores))
        trial_record = {
            "trial": int(len(trials_history) + 1),
            "params": {k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in params.items()},
            "cv_roc_auc": round(float(mean_score), 4)
        }
        trials_history.append(trial_record)
        print(f"  Trial {trial_record['trial']:>2}/{n_calls} | CV ROC-AUC: {mean_score:.4f} | params: {trial_record['params']}")
        return -mean_score  # Minimize negative AUC
        
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    
    best_params = {
        "learning_rate": float(res_gp.x[0]),
        "max_iter": int(res_gp.x[1]),
        "max_depth": int(res_gp.x[2]),
        "min_samples_leaf": int(res_gp.x[3]),
        "l2_regularization": float(res_gp.x[4])
    }
    print("\nBest Hyperparameters Found by Bayesian Optimization:")
    print(json.dumps(best_params, indent=2))
    
    # 3. Refit Tuned Model on full training data
    print("\n[3/3] Refitting Tuned Model and Computing Evaluation Comparison...")
    tuned_clf = HistGradientBoostingClassifier(**best_params, random_state=42)
    tuned_clf.fit(X_train, y_train)
    
    t_inf1 = time.time()
    tuned_probs = tuned_clf.predict_proba(X_full)[:, 1]
    tuned_inf_time = round((time.time() - t_inf1) / len(df_clean) * 1000, 4)
    df_clean["pred_prob"] = tuned_probs
    tuned_top1 = compute_top1_acc(df_clean, "pred_prob")
    tuned_loss = float(log_loss(y_full, tuned_probs))
    tuned_f1 = float(f1_score(y_full, (tuned_probs >= 0.5).astype(int), zero_division=0))
    tuned_auc = float(roc_auc_score(y_full, tuned_probs))
    
    # Save tuned pipeline
    from sklearn.pipeline import Pipeline
    tuned_pipeline = Pipeline([
        ("features", feat_pipe),
        ("clf", tuned_clf)
    ])
    joblib.dump(tuned_pipeline, mod_path / "tuned_gradient_boosting.joblib")
    print(f"Saved tuned model to {mod_path / 'tuned_gradient_boosting.joblib'}")
    
    report = {
        "model": "HistGradientBoostingClassifier",
        "optimisation_strategy": "Bayesian Optimization with Gaussian Process (gp_minimize)",
        "search_space": {
            "learning_rate": [0.02, 0.20],
            "max_iter": [50, 150],
            "max_depth": [4, 12],
            "min_samples_leaf": [10, 40],
            "l2_regularization": [0.0001, 5.0]
        },
        "best_hyperparameters": best_params,
        "comparison": {
            "untuned": {
                "top_1_accuracy": round(untuned_top1, 4),
                "f1_score": round(untuned_f1, 4),
                "roc_auc": round(untuned_auc, 4),
                "log_loss": round(untuned_loss, 4),
                "inference_time_ms": untuned_inf_time
            },
            "tuned": {
                "top_1_accuracy": round(tuned_top1, 4),
                "f1_score": round(tuned_f1, 4),
                "roc_auc": round(tuned_auc, 4),
                "log_loss": round(tuned_loss, 4),
                "inference_time_ms": tuned_inf_time
            },
            "improvement": {
                "top_1_acc_delta": f"+{(tuned_top1 - untuned_top1)*100:.2f}%",
                "log_loss_reduction": f"{untuned_loss - tuned_loss:.4f}"
            }
        },
        "trials": trials_history
    }
    
    def default_serializer(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        return str(o)
        
    report_file = res_path / "bayesian_optimization_report.json"
    report_file.write_text(json.dumps(report, indent=2, default=default_serializer))
    print(f"Saved Bayesian optimization report to {report_file}")
    
    print("\n" + "=" * 70)
    print("📈 BAYESIAN OPTIMIZATION RESULT SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Untuned Model':<15} | {'Tuned Model':<15} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'Top-1 Accuracy':<25} | {untuned_top1*100:>13.2f}% | {tuned_top1*100:>13.2f}% | {report['comparison']['improvement']['top_1_acc_delta']}")
    print(f"{'F1-Score':<25} | {untuned_f1:>15.4f} | {tuned_f1:>15.4f} | {tuned_f1 - untuned_f1:+.4f}")
    print(f"{'ROC-AUC':<25} | {untuned_auc:>15.4f} | {tuned_auc:>15.4f} | {tuned_auc - untuned_auc:+.4f}")
    print(f"{'Log Loss':<25} | {untuned_loss:>15.4f} | {tuned_loss:>15.4f} | {untuned_loss - tuned_loss:+.4f}")
    print(f"{'Inference Time':<25} | {untuned_inf_time:>12.3f} ms | {tuned_inf_time:>12.3f} ms |")
    print("=" * 70 + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Hyperparameter Optimization")
    parser.add_argument("--candidates", default="results/train_candidate_sample_k64.csv")
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    run_bayesian_optimization(args.candidates, n_calls=args.trials)


if __name__ == "__main__":
    main()
