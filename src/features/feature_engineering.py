"""Candidate feature engineering for map matching classification and ranking."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def circular_heading_difference(theta1, theta2):
    """Compute shortest angular difference between two angles in radians [-pi, pi].
    
    Result is always in [0, pi].
    """
    diff = np.abs(theta1 - theta2) % (2.0 * np.pi)
    return np.minimum(diff, 2.0 * np.pi - diff)


def engineer_candidate_features(df, sigma_gps=15.0):
    """Compute discriminative spatial, angular, and kinematic features from candidate rows.
    
    Args:
        df: DataFrame conforming to candidate schema from candidate_generation.py.
        sigma_gps: Assumed GPS noise standard deviation in metres (default: 15.0).
        
    Returns:
        DataFrame with added feature columns.
    """
    out = df.copy()
    
    # 1. Non-linear distance transformations
    d = np.maximum(out["distance_m"].to_numpy(dtype=float), 0.0)
    out["log_distance_m"] = np.log1p(d)
    out["inv_distance_m"] = 1.0 / (1.0 + d)
    
    # Midpoint distance non-linear features
    m_dist = np.maximum(out["midpoint_distance_m"].to_numpy(dtype=float), 0.0)
    out["log_midpoint_dist"] = np.log1p(m_dist)
    
    # 2. Gaussian emission probability proxy (canonical HMM emission term)
    out["emission_prob_proxy"] = np.exp(-0.5 * (d / sigma_gps) ** 2)
    
    # 3. Angular and bearing alignment
    arc_bearing = out["bearing_rad"].to_numpy(dtype=float)
    gps_heading = out["gps_heading_rad"].to_numpy(dtype=float)
    
    heading_diff = circular_heading_difference(arc_bearing, gps_heading)
    out["heading_diff_rad"] = heading_diff
    out["cos_heading_diff"] = np.cos(arc_bearing - gps_heading)
    
    # 4. Projection geometry features
    frac = np.clip(out["projection_fraction"].to_numpy(dtype=float), 0.0, 1.0)
    out["is_endpoint_projection"] = ((frac <= 1e-4) | (frac >= 0.9999)).astype(float)
    out["projection_center_dev"] = np.abs(frac - 0.5)
    
    # 5. Segment attributes
    seg_len = np.maximum(out["segment_length_m"].to_numpy(dtype=float), 0.0)
    out["log_segment_length_m"] = np.log1p(seg_len)
    
    # 6. Kinematic interactions
    speed = np.maximum(out["gps_speed_mps"].to_numpy(dtype=float), 0.0)
    out["speed_alignment_interaction"] = speed * out["cos_heading_diff"]
    
    # 7. Within-observation relative ranking features
    # (How much better is this candidate compared to the best candidate for the same GPS point?)
    grouped = out.groupby(["trip_id", "observation_idx"], sort=False)["distance_m"]
    min_dist_per_obs = grouped.transform("min")
    out["dist_diff_from_min"] = d - min_dist_per_obs
    out["dist_ratio_to_min"] = (d + 1.0) / (min_dist_per_obs + 1.0)
    out["cand_distance_rank"] = grouped.rank(method="dense", ascending=True)
    
    return out


# Canonical list of engineered feature names used for model inputs
FEATURE_COLUMNS = [
    "distance_m",
    "log_distance_m",
    "inv_distance_m",
    "midpoint_distance_m",
    "log_midpoint_dist",
    "emission_prob_proxy",
    "heading_diff_rad",
    "cos_heading_diff",
    "projection_fraction",
    "is_endpoint_projection",
    "projection_center_dev",
    "segment_length_m",
    "log_segment_length_m",
    "gps_speed_mps",
    "speed_alignment_interaction",
    "time_gap_s",
    "dist_diff_from_min",
    "dist_ratio_to_min",
    "cand_distance_rank",
]


class CandidateFeaturePipeline(BaseEstimator, TransformerMixin):
    """End-to-end scikit-learn transformer for candidate feature extraction and standardization."""
    
    def __init__(self, feature_columns=None, sigma_gps=15.0):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.sigma_gps = sigma_gps
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame conforming to candidate schema")
        df_feat = engineer_candidate_features(X, sigma_gps=self.sigma_gps)
        feat_matrix = df_feat[self.feature_columns].fillna(0.0).to_numpy(dtype=float)
        self.scaler.fit(feat_matrix)
        return self
        
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame conforming to candidate schema")
        df_feat = engineer_candidate_features(X, sigma_gps=self.sigma_gps)
        feat_matrix = df_feat[self.feature_columns].fillna(0.0).to_numpy(dtype=float)
        return self.scaler.transform(feat_matrix)
        
    def get_feature_names(self):
        return list(self.feature_columns)
