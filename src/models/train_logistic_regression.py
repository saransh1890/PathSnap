"""Train and evaluate Logistic Regression for map-matching candidate classification."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_engineering import CandidateFeaturePipeline, FEATURE_COLUMNS


def compute_top_k_accuracy(df_eval, prob_col="pred_prob", k_vals=(1, 3, 5)):
    """Compute Top-K candidate accuracy per observation.
    
    Checks if the true reference arc is ranked in the top K candidates by predicted probability.
    """
    results = {}
    grouped = df_eval.groupby(["trip_id", "observation_idx"], sort=False)
    
    total_obs = len(grouped)
    if total_obs == 0:
        return {f"top_{k}_acc": 0.0 for k in k_vals}
        
    for k in k_vals:
        correct = 0
        for _, group in grouped:
            top_cands = group.nlargest(k, prob_col)
            if (top_cands["label"] == 1).any():
                correct += 1
        results[f"top_{k}_acc"] = float(correct / total_obs)
        
    return results


def train_logistic_regression(df, c_values=(0.01, 0.1, 1.0, 10.0), n_splits=5):
    """Train Logistic Regression with GroupKFold cross-validation on trips."""
    # Filter only labeled training rows
    df_train = df[df["label"].isin([0, 1])].copy()
    groups = df_train["trip_id"].to_numpy()
    y = df_train["label"].to_numpy(dtype=int)
    
    print(f"Dataset: {len(df_train)} rows across {df_train['trip_id'].nunique()} trips")
    print(f"Class distribution: Positives={np.sum(y == 1)}, Negatives={np.sum(y == 0)}")
    
    gkf = GroupKFold(n_splits=n_splits)
    best_c = 1.0
    best_score = -1.0
    cv_records = []
    
    for c in c_values:
        fold_aucs = []
        fold_top1s = []
        fold_aps = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, y, groups)):
            train_sub = df_train.iloc[train_idx]
            val_sub = df_train.iloc[val_idx].copy()
            
            pipeline = Pipeline([
                ("features", CandidateFeaturePipeline()),
                ("clf", LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42
                ))
            ])
            
            pipeline.fit(train_sub, train_sub["label"])
            val_probs = pipeline.predict_proba(val_sub)[:, 1]
            val_sub["pred_prob"] = val_probs
            
            auc = roc_auc_score(val_sub["label"], val_probs)
            ap = average_precision_score(val_sub["label"], val_probs)
            top1 = compute_top_k_accuracy(val_sub, prob_col="pred_prob", k_vals=(1,))["top_1_acc"]
            
            fold_aucs.append(auc)
            fold_aps.append(ap)
            fold_top1s.append(top1)
            
        mean_auc = float(np.mean(fold_aucs))
        mean_ap = float(np.mean(fold_aps))
        mean_top1 = float(np.mean(fold_top1s))
        
        print(f"C={c:<6} | CV ROC-AUC: {mean_auc:.4f} | CV PR-AUC: {mean_ap:.4f} | CV Top-1 Acc: {mean_top1*100:.2f}%")
        cv_records.append({
            "C": c, "mean_auc": mean_auc, "mean_ap": mean_ap, "mean_top1_acc": mean_top1
        })
        
        # Select best model based on Top-1 ranking accuracy and ROC-AUC
        score = mean_top1 + 0.1 * mean_auc
        if score > best_score:
            best_score = score
            best_c = c
            
    print(f"\nOptimal C selected: {best_c}")
    
    # Train final model on full training set
    final_pipeline = Pipeline([
        ("features", CandidateFeaturePipeline()),
        ("clf", LogisticRegression(
            C=best_c,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ))
    ])
    final_pipeline.fit(df_train, y)
    
    # Feature importances / coefficients
    coefs = final_pipeline.named_steps["clf"].coef_[0]
    feat_names = final_pipeline.named_steps["features"].get_feature_names()
    coef_dict = {name: float(coef) for name, coef in zip(feat_names, coefs)}
    
    # Evaluate final training fit
    final_probs = final_pipeline.predict_proba(df_train)[:, 1]
    df_train["pred_prob"] = final_probs
    top_metrics = compute_top_k_accuracy(df_train, prob_col="pred_prob", k_vals=(1, 3, 5))
    train_auc = float(roc_auc_score(y, final_probs))
    train_ap = float(average_precision_score(y, final_probs))
    
    summary = {
        "model": "LogisticRegression",
        "best_C": best_c,
        "train_roc_auc": train_auc,
        "train_pr_auc": train_ap,
        "train_top_1_acc": top_metrics["top_1_acc"],
        "train_top_3_acc": top_metrics["top_3_acc"],
        "train_top_5_acc": top_metrics["top_5_acc"],
        "coefficients": coef_dict,
        "cv_results": cv_records
    }
    
    return final_pipeline, summary


def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression Candidate Ranker")
    parser.add_argument("--candidates", default="results/train_candidate_sample_k64.csv",
                        help="Path to candidate CSV")
    parser.add_argument("--out-dir", default="models", help="Directory to save model artifacts")
    parser.add_argument("--results-dir", default="results", help="Directory to save metrics")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    res_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading candidates from {args.candidates}...")
    df = pd.read_csv(args.candidates)
    
    pipeline, summary = train_logistic_regression(df)
    
    # Save model artifact
    model_path = out_dir / "logistic_regression.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save metrics JSON
    metrics_path = res_dir / "logistic_regression_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")
    
    print("\nTop 5 positive predictive coefficients:")
    sorted_pos = sorted(summary["coefficients"].items(), key=lambda x: x[1], reverse=True)[:5]
    for k, v in sorted_pos:
        print(f"  {k:<30}: {v:+.4f}")
        
    print("\nTop 5 negative predictive coefficients:")
    sorted_neg = sorted(summary["coefficients"].items(), key=lambda x: x[1])[:5]
    for k, v in sorted_neg:
        print(f"  {k:<30}: {v:+.4f}")
        
    print(f"\nFinal Train Top-1 Accuracy: {summary['train_top_1_acc']*100:.2f}%")
    print(f"Final Train Top-3 Accuracy: {summary['train_top_3_acc']*100:.2f}%")
    print(f"Final Train ROC-AUC:        {summary['train_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
