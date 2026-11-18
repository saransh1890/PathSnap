"""Generate publication-quality evaluation charts and plots for the technical report."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_evaluation_plots(results_dir="results", out_dir="results/plots"):
    res_path = Path(results_dir)
    plots_path = Path(out_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Load 5-model benchmark summary
    summary_file = res_path / "all_models_benchmark_summary.csv"
    if not summary_file.exists():
        print(f"Warning: {summary_file} not found.")
        return
        
    df = pd.read_csv(summary_file)
    print(f"Generating benchmark plots for {len(df)} models...")
    
    # Set plot styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------
    # 1. Top-1 vs Top-3 Accuracy Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df["top_1_accuracy"] * 100, width, label="Top-1 Accuracy (%)", color="#1f77b4")
    rects2 = ax.bar(x + width/2, df["top_3_accuracy"] * 100, width, label="Top-3 Accuracy (%)", color="#2ca02c")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("PathSnap Model Comparison: Top-1 vs Top-3 Ranking Accuracy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", frameon=True)
    
    # Add data labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                    
    plt.tight_layout()
    chart1_path = plots_path / "model_accuracy_comparison.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved: {chart1_path}")
    
    # -------------------------------------------------------------
    # 2. Cross-Validation Macro F1 vs ROC-AUC
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects3 = ax.bar(x - width/2, df["cv_macro_f1"], width, label="CV Macro F1-Score", color="#ff7f0e")
    rects4 = ax.bar(x + width/2, df["roc_auc"], width, label="ROC-AUC", color="#9467bd")
    
    ax.set_ylabel("Score (0 to 1)", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Validation Macro F1-Score and ROC-AUC Across 5 Model Families", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", frameon=True)
    
    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects4:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                    
    plt.tight_layout()
    chart2_path = plots_path / "cv_f1_and_auc_comparison.png"
    plt.savefig(chart2_path)
    plt.close()
    print(f"Saved: {chart2_path}")
    
    # -------------------------------------------------------------
    # 3. Log Loss & Computational Inference Latency
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    # Log loss
    ax1.bar(df["model"], df["log_loss"], color="#d62728", width=0.5)
    ax1.set_ylabel("Log Loss (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Calibration: Log Loss", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(df["model"], rotation=25, ha="right", fontsize=9)
    for i, v in enumerate(df["log_loss"]):
        ax1.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        
    # Inference Latency
    ax2.bar(df["model"], df["inference_time_ms"], color="#8c564b", width=0.5)
    ax2.set_ylabel("Inference Time per Sample (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Computational Efficiency: Latency (ms)", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(df["model"], rotation=25, ha="right", fontsize=9)
    for i, v in enumerate(df["inference_time_ms"]):
        ax2.text(i, v + 0.0005, f"{v:.3f} ms", ha="center", va="bottom", fontsize=9)
        
    plt.tight_layout()
    chart3_path = plots_path / "logloss_and_latency_comparison.png"
    plt.savefig(chart3_path)
    plt.close()
    print(f"Saved: {chart3_path}")
    
    # -------------------------------------------------------------
    # 4. Bayesian Optimization Trial Progression (if available)
    # -------------------------------------------------------------
    bayes_file = res_path / "bayesian_optimization_report.json"
    if bayes_file.exists():
        bayes_data = json.loads(bayes_file.read_text())
        trials = bayes_data.get("trials", [])
        if trials:
            trial_nums = [t["trial"] for t in trials]
            scores = [t["cv_roc_auc"] for t in trials]
            
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.plot(trial_nums, scores, marker="o", color="#1f77b4", linewidth=2, markersize=7)
            best_trial_idx = np.argmax(scores)
            ax.plot(trial_nums[best_trial_idx], scores[best_trial_idx], marker="*", color="red", markersize=14, label="Best Configuration")
            
            ax.set_xlabel("Optimization Trial", fontsize=11, fontweight="bold")
            ax.set_ylabel("Cross-Validation ROC-AUC", fontsize=11, fontweight="bold")
            ax.set_title("Gaussian Process Bayesian Hyperparameter Optimization Progression", fontsize=12, fontweight="bold")
            ax.set_xticks(trial_nums)
            ax.legend(loc="lower right")
            
            plt.tight_layout()
            chart4_path = plots_path / "bayesian_optimization_trials.png"
            plt.savefig(chart4_path)
            plt.close()
            print(f"Saved: {chart4_path}")


if __name__ == "__main__":
    generate_evaluation_plots()
