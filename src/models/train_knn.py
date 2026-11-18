"""Train and evaluate K-Nearest Neighbours (KNN) for candidate classification and ranking."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_engineering import CandidateFeaturePipeline


def compute_top_k_accuracy(df_eval, prob_col="pred_prob", k_vals=(1, 3, 5)):
    """Compute Top-K candidate ranking accuracy per observation."""
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


def sample_for_knn(df, neg_ratio=5, random_state=42):
    """Subsample negative candidates to keep balanced high-informative dataset for fast KNN.
    
    Keeps all positive candidates and the top `neg_ratio` closest negative candidates per observation.
    """
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    
    # Select closest negatives per observation (hard negative mining)
    closest_neg = (
        negatives.sort_values(["trip_id", "observation_idx", "distance_m"])
        .groupby(["trip_id", "observation_idx"], sort=False)
        .head(neg_ratio)
    )
    
    sampled = pd.concat([positives, closest_neg], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_knn(df, k_neighbors_list=(5, 11, 21), n_splits=5, neg_ratio=5):
    """Train KNN with GroupKFold cross-validation on trips."""
    df_clean = df[df["label"].isin([0, 1])].copy()
    
    print(f"Original dataset: {len(df_clean)} rows across {df_clean['trip_id'].nunique()} trips")
    df_train = sample_for_knn(df_clean, neg_ratio=neg_ratio)
    print(f"KNN training subset (hard negatives): {len(df_train)} rows (Positives={sum(df_train['label']==1)}, Negatives={sum(df_train['label']==0)})")
    
    groups = df_train["trip_id"].to_numpy()
    y = df_train["label"].to_numpy(dtype=int)
    
    gkf = GroupKFold(n_splits=n_splits)
    best_k = 11
    best_score = -1.0
    cv_records = []
    
    for k in k_neighbors_list:
        fold_aucs = []
        fold_aps = []
        fold_top1s = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, y, groups)):
            train_sub = df_train.iloc[train_idx]
            val_sub = df_train.iloc[val_idx].copy()
            
            pipeline = Pipeline([
                ("features", CandidateFeaturePipeline()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=k,
                    weights="distance",
                    algorithm="auto",
                    n_jobs=-1
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
        
        print(f"k_neighbors={k:<4} | CV ROC-AUC: {mean_auc:.4f} | CV PR-AUC: {mean_ap:.4f} | CV Top-1 Acc: {mean_top1*100:.2f}%")
        cv_records.append({
            "k_neighbors": k, "mean_auc": mean_auc, "mean_ap": mean_ap, "mean_top1_acc": mean_top1
        })
        
        score = mean_top1 + 0.1 * mean_auc
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\nOptimal K selected: {best_k}")
    
    # Train final model on the full sampled dataset
    final_pipeline = Pipeline([
        ("features", CandidateFeaturePipeline()),
        ("clf", KNeighborsClassifier(
            n_neighbors=best_k,
            weights="distance",
            algorithm="auto",
            n_jobs=-1
        ))
    ])
    final_pipeline.fit(df_train, y)
    
    # Evaluate final model predictions
    final_probs = final_pipeline.predict_proba(df_train)[:, 1]
    df_train["pred_prob"] = final_probs
    top_metrics = compute_top_k_accuracy(df_train, prob_col="pred_prob", k_vals=(1, 3, 5))
    train_auc = float(roc_auc_score(y, final_probs))
    train_ap = float(average_precision_score(y, final_probs))
    
    summary = {
        "model": "KNeighborsClassifier",
        "best_k_neighbors": best_k,
        "train_roc_auc": train_auc,
        "train_pr_auc": train_ap,
        "train_top_1_acc": top_metrics["top_1_acc"],
        "train_top_3_acc": top_metrics["top_3_acc"],
        "train_top_5_acc": top_metrics["top_5_acc"],
        "cv_results": cv_records
    }
    
    return final_pipeline, summary


def main():
    parser = argparse.ArgumentParser(description="Train KNN Candidate Ranker")
    parser.add_argument("--candidates", default="results/train_candidate_sample_k64.csv",
                        help="Path to candidate CSV")
    parser.add_argument("--out-dir", default="models", help="Directory to save model artifacts")
    parser.add_argument("--results-dir", default="results", help="Directory to save metrics")
    parser.add_argument("--neg-ratio", type=int, default=5, help="Number of hard negatives per observation")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    res_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading candidates from {args.candidates}...")
    df = pd.read_csv(args.candidates)
    
    pipeline, summary = train_knn(df, neg_ratio=args.neg_ratio)
    
    # Save model artifact
    model_path = out_dir / "knn.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save metrics JSON
    metrics_path = res_dir / "knn_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")
    
    print(f"\nFinal Train Top-1 Accuracy: {summary['train_top_1_acc']*100:.2f}%")
    print(f"Final Train Top-3 Accuracy: {summary['train_top_3_acc']*100:.2f}%")
    print(f"Final Train ROC-AUC:        {summary['train_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
