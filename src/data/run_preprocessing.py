"""Batch GPS data preprocessing runner for PathSnap."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import iter_trips
from data.preprocessing import (
    project_lonlat, prepare_track, filter_gps_spikes,
    smooth_trajectory_xy, compute_kinematics
)


def preprocess_trip(trip_dict, smooth=False, max_speed_mps=50.0):
    """Run full GPS preprocessing pipeline on one trip dictionary."""
    trip_id = trip_dict["trip_id"]
    track = trip_dict["track"]
    
    # 1. Coordinate validity & timestamp sanity
    prep = prepare_track(track, sort_by_time=True)
    clean_track = prep["track"]
    valid_mask = prep["valid_mask"]
    
    # 2. Local metric Cartesian projection
    pts_valid = clean_track[valid_mask, :2]
    ts_valid = clean_track[valid_mask, 2]
    
    if len(pts_valid) > 0:
        xy_proj, origin = project_lonlat(pts_valid)
    else:
        xy_proj = np.empty((0, 2), dtype=float)
        origin = (0.0, 0.0)
        
    # 3. GPS jump / spike detection
    spike_keep = filter_gps_spikes(xy_proj, ts_valid, max_speed_mps=max_speed_mps)
    
    # 4. Trajectory smoothing (optional)
    xy_final = smooth_trajectory_xy(xy_proj, alpha=0.3) if smooth else xy_proj
    
    # 5. Kinematic computation
    kinematics = compute_kinematics(xy_final, ts_valid)
    
    stats = {
        "trip_id": trip_id,
        "total_points": len(track),
        "valid_points": int(valid_mask.sum()),
        "invalid_points": int((~valid_mask).sum()),
        "spikes_flagged": int((~spike_keep).sum()),
        "nonincreasing_timestamps": prep["nonincreasing_steps"],
        "gaps_over_10s": prep["gaps_over_10s"],
        "max_speed_observed_mps": float(np.max(kinematics["speeds_mps"])) if len(kinematics["speeds_mps"]) else 0.0,
        "mean_speed_mps": float(np.mean(kinematics["speeds_mps"])) if len(kinematics["speeds_mps"]) else 0.0,
        "total_distance_m": float(np.sum(kinematics["distances_m"])) if len(kinematics["distances_m"]) else 0.0,
        "origin_lon": origin[0],
        "origin_lat": origin[1]
    }
    
    return stats


def run_batch_preprocessing(dataset_root, out_dir="results/preprocessed", smooth=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    for partition in ["train", "student_test"]:
        part_dir = Path(dataset_root) / partition
        if not part_dir.exists():
            continue
        print(f"Preprocessing partition '{partition}'...")
        for trip_dict in iter_trips(dataset_root, partition):
            stats = preprocess_trip(trip_dict, smooth=smooth)
            stats["partition"] = partition
            all_stats.append(stats)
            
    df_stats = pd.DataFrame(all_stats)
    summary_path = out / "preprocessing_summary.csv"
    df_stats.to_csv(summary_path, index=False)
    print(f"\nSaved preprocessing summary to {summary_path}")
    
    summary_metrics = {
        "total_trips": len(df_stats),
        "total_gps_points": int(df_stats["total_points"].sum()),
        "total_invalid_points": int(df_stats["invalid_points"].sum()),
        "total_spikes_detected": int(df_stats["spikes_flagged"].sum()),
        "trips_with_large_gaps": int((df_stats["gaps_over_10s"] > 0).sum()),
        "mean_trip_distance_m": float(df_stats["total_distance_m"].mean()),
        "mean_speed_mps": float(df_stats["mean_speed_mps"].mean())
    }
    
    metrics_path = out / "preprocessing_metrics.json"
    metrics_path.write_text(json.dumps(summary_metrics, indent=2))
    print(f"Saved preprocessing report to {metrics_path}")
    print("\nBatch Preprocessing Overview:")
    print(json.dumps(summary_metrics, indent=2))
    return df_stats


def main():
    parser = argparse.ArgumentParser(description="Batch GPS Preprocessing for PathSnap")
    parser.add_argument("--root", default="data", help="Dataset root containing train/ and student_test/")
    parser.add_argument("--out", default="results/preprocessed", help="Output directory for reports")
    parser.add_argument("--smooth", action="store_true", help="Apply exponential smoothing")
    args = parser.parse_args()
    
    run_batch_preprocessing(args.root, out_dir=args.out, smooth=args.smooth)


if __name__ == "__main__":
    main()
