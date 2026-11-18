"""Hidden Markov Model (HMM) and Viterbi sequential route decoding for PathSnap."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import load_trip
from graph.road_graph import RoadGraph
from candidates.candidate_generation import generate_candidates


class HMMRouteDecoder:
    """HMM sequence decoder combining classifier emission probabilities with road-network topology.
    
    Attributes:
        road_graph: RoadGraph instance containing road topology and shortest path matrix.
        beta: Scaling parameter for transition probability (default: 15.0 m).
        sigma_z: Standard deviation for Gaussian GPS emission probability (default: 15.0 m).
    """
    
    def __init__(self, road_graph, beta=15.0, sigma_z=15.0):
        self.road_graph = road_graph
        self.beta = beta
        self.sigma_z = sigma_z
        
    def compute_emission_log_prob(self, cands_group, prob_col="pred_prob"):
        """Compute log emission probabilities for a set of candidate arcs for one GPS point."""
        if prob_col in cands_group.columns:
            probs = np.clip(cands_group[prob_col].to_numpy(dtype=float), 1e-6, 1.0)
            return np.log(probs)
        else:
            # Fallback to Gaussian emission based on orthogonal distance_m
            d = cands_group["distance_m"].to_numpy(dtype=float)
            return -0.5 * (d / self.sigma_z) ** 2 - np.log(np.sqrt(2 * np.pi) * self.sigma_z)
            
    def compute_transition_log_prob(self, arc_u, arc_v, gps_dist_m):
        """Compute log transition probability from arc_u to arc_v given observed GPS displacement.
        
        P(arc_v | arc_u) = 1/beta * exp(-|gps_dist - net_dist| / beta)
        """
        if arc_u == arc_v:
            # Vehicle stayed on the same road arc
            net_dist = 0.0
        else:
            net_dist = self.road_graph.network_transition_distance(arc_u, arc_v)
            
        if np.isinf(net_dist) or net_dist > 5000.0:
            # Unreachable in network: large penalty log-prob
            return -20.0
            
        diff = np.abs(gps_dist_m - net_dist)
        return - (diff / self.beta) - np.log(self.beta)
        
    def decode(self, cands_df, prob_col="pred_prob", max_cands_per_step=16):
        """Run Viterbi decoding over candidate observation sequence.
        
        Args:
            cands_df: DataFrame of candidate rows sorted by observation_idx.
            prob_col: Column with candidate emission/match probabilities.
            max_cands_per_step: Prune to top-N candidates per step for fast decoding.
            
        Returns:
            list of dicts with observation_idx, chosen_arc_id, match_prob, and continuity flag.
        """
        grouped = [
            (obs_idx, group.nlargest(max_cands_per_step, prob_col if prob_col in group.columns else "distance_m"))
            for obs_idx, group in cands_df.groupby("observation_idx", sort=True)
        ]
        
        T = len(grouped)
        if T == 0:
            return []
            
        # Viterbi DP structures
        # V[t][j] = log-prob of best path ending at candidate j at step t
        # backpointer[t][j] = index of best predecessor candidate at step t-1
        
        # Step 0: Initialization
        obs0_idx, cands0 = grouped[0]
        arcs_0 = cands0["arc_id"].to_numpy(dtype=int)
        v_current = self.compute_emission_log_prob(cands0, prob_col)
        
        backpointers = []
        arcs_history = [arcs_0]
        cands_history = [cands0]
        
        # Steps 1 to T-1
        for t in range(1, T):
            prev_obs_idx, prev_cands = grouped[t - 1]
            curr_obs_idx, curr_cands = grouped[t]
            
            prev_arcs = arcs_history[-1]
            curr_arcs = curr_cands["arc_id"].to_numpy(dtype=int)
            arcs_history.append(curr_arcs)
            cands_history.append(curr_cands)
            
            # Approximate GPS distance between consecutive observations
            gps_speed = float(curr_cands["gps_speed_mps"].iloc[0]) if "gps_speed_mps" in curr_cands.columns else 10.0
            time_gap = float(curr_cands["time_gap_s"].iloc[0]) if "time_gap_s" in curr_cands.columns else 1.0
            gps_step_m = max(gps_speed * time_gap, 1.0)
            
            log_emissions = self.compute_emission_log_prob(curr_cands, prob_col)
            
            v_next = np.full(len(curr_arcs), -np.inf)
            bp_t = np.zeros(len(curr_arcs), dtype=int)
            
            # Transition matrix from all prev_arcs to all curr_arcs
            for j, arc_v in enumerate(curr_arcs):
                best_val = -np.inf
                best_i = 0
                for i, arc_u in enumerate(prev_arcs):
                    trans_lp = self.compute_transition_log_prob(arc_u, arc_v, gps_step_m)
                    val = v_current[i] + trans_lp
                    if val > best_val:
                        best_val = val
                        best_i = i
                v_next[j] = best_val + log_emissions[j]
                bp_t[j] = best_i
                
            # If all paths underflowed (e.g. disconnected components), fallback to emission only
            if np.all(np.isneginf(v_next)):
                v_next = log_emissions
                bp_t = np.argmax(v_current) * np.ones(len(curr_arcs), dtype=int)
                
            backpointers.append(bp_t)
            v_current = v_next
            
        # Backtracking
        best_last_idx = int(np.argmax(v_current))
        best_path_indices = [best_last_idx]
        
        for t in range(T - 2, -1, -1):
            prev_idx = backpointers[t][best_path_indices[-1]]
            best_path_indices.append(prev_idx)
            
        best_path_indices.reverse()
        
        # Build decoded route output
        decoded_route = []
        for t in range(T):
            obs_idx, cands = grouped[t]
            chosen_cand_idx = best_path_indices[t]
            chosen_row = cands.iloc[chosen_cand_idx]
            
            decoded_route.append({
                "observation_idx": int(obs_idx),
                "chosen_arc_id": int(chosen_row["arc_id"]),
                "reference_arc_id": int(chosen_row["reference_arc_id"]) if "reference_arc_id" in chosen_row else -1,
                "label": int(chosen_row["label"]) if "label" in chosen_row else -1,
                "distance_m": float(chosen_row["distance_m"]),
                "confidence_prob": float(chosen_row[prob_col]) if prob_col in chosen_row else 1.0
            })
            
        return decoded_route


def evaluate_route_continuity(road_graph, route_arc_ids):
    """Compute Route Continuity percentage: ratio of contiguous/connected consecutive arcs."""
    if len(route_arc_ids) <= 1:
        return 1.0
        
    connected_steps = 0
    total_steps = len(route_arc_ids) - 1
    
    for t in range(total_steps):
        u, v = route_arc_ids[t], route_arc_ids[t + 1]
        if u == v or road_graph.are_arcs_connected(u, v):
            connected_steps += 1
        elif road_graph.network_transition_distance(u, v) < 100.0:
            connected_steps += 1
            
    return float(connected_steps / total_steps)


def compare_classifier_vs_hmm(trip_folder, model_path, k=8, beta=15.0, max_points=100):
    """Run head-to-head comparison: Classifier-Only vs Classifier + HMM."""
    d = load_trip(trip_folder)
    road_graph = RoadGraph.from_trip_dict(d)
    
    # Generate candidates with optional sampling for speed
    cands, cov = generate_candidates(trip_folder, k=k, include_reference=True, max_points=max_points)
    if len(cands) == 0:
        return None
        
    # Score candidates using ML classifier
    pipeline = joblib.load(model_path)
    cands["pred_prob"] = pipeline.predict_proba(cands)[:, 1]
    
    # 1. Classifier-Only Top-1 selection (greedy per observation)
    clf_top1 = cands.sort_values("pred_prob", ascending=False).groupby("observation_idx", sort=True).first().reset_index()
    clf_arcs = clf_top1["arc_id"].to_list()
    clf_acc = float((clf_top1["label"] == 1).mean()) if "label" in clf_top1 else 0.0
    clf_continuity = evaluate_route_continuity(road_graph, clf_arcs)
    
    # 2. HMM + Viterbi Global Decoding
    decoder = HMMRouteDecoder(road_graph, beta=beta)
    hmm_decoded = decoder.decode(cands, prob_col="pred_prob", max_cands_per_step=6)
    df_hmm = pd.DataFrame(hmm_decoded)
    hmm_arcs = df_hmm["chosen_arc_id"].to_list()
    hmm_acc = float((df_hmm["label"] == 1).mean()) if "label" in df_hmm else 0.0
    hmm_continuity = evaluate_route_continuity(road_graph, hmm_arcs)
    
    return {
        "trip_id": d["trip_id"],
        "num_observations": len(clf_arcs),
        "classifier_only_acc": round(clf_acc, 4),
        "hmm_viterbi_acc": round(hmm_acc, 4),
        "classifier_continuity": round(clf_continuity, 4),
        "hmm_continuity": round(hmm_continuity, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="HMM Viterbi Route Reconstruction Runner")
    parser.add_argument("--trip", default="data/train/00000001", help="Path to trip folder")
    parser.add_argument("--model", default="models/logistic_regression.joblib", help="Classifier pipeline path")
    parser.add_argument("--k", type=int, default=16, help="Candidates per observation")
    args = parser.parse_args()
    
    print(f"Comparing Classifier-Only vs Classifier + HMM for {args.trip}...")
    res = compare_classifier_vs_hmm(args.trip, args.model, k=args.k)
    
    print("\n" + "=" * 70)
    print("🛣️  HMM & VITERBI ROUTE RECONSTRUCTION COMPARISON")
    print("=" * 70)
    print(f"Trip ID:                     {res['trip_id']}")
    print(f"Total GPS Observations:      {res['num_observations']}")
    print(f"Classifier-Only Top-1 Acc:   {res['classifier_only_acc']*100:.2f}%")
    print(f"HMM + Viterbi Decoded Acc:   {res['hmm_viterbi_acc']*100:.2f}%")
    print(f"Classifier Route Continuity: {res['classifier_continuity']*100:.2f}%")
    print(f"HMM Route Continuity:        {res['hmm_continuity']*100:.2f}%  (🚀 Enforces physical road network)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
