"""FastAPI backend service for PathSnap distributed evaluation (Laptop 1)."""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.loader import load_trip
from graph.road_graph import RoadGraph
from candidates.candidate_generation import generate_candidates
from hmm.route_reconstruction import HMMRouteDecoder, evaluate_route_continuity

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed Map-Matching and Route Reconstruction Backend Service",
    version="1.0.0"
)

# Enable CORS for Two-Laptop deployment across LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "logistic_regression.joblib"
_cached_model = None


def get_model():
    global _cached_model
    if _cached_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        _cached_model = joblib.load(MODEL_PATH)
    return _cached_model


@app.get("/")
def health_check():
    """Health check endpoint for Laptop 2 connectivity."""
    return {
        "status": "online",
        "service": "PathSnap Map-Matching API",
        "version": "1.0.0",
        "architecture": "Two-Laptop Deployment (Laptop 1: Model & API Host)"
    }


@app.get("/api/trips")
def list_available_trips():
    """Return list of available reserved evaluation and testing trips."""
    trips = []
    # Check student_test and train partitions
    for partition in ["student_test", "train"]:
        part_dir = DATA_ROOT / partition
        if part_dir.exists():
            for folder in sorted(part_dir.iterdir()):
                if folder.is_dir():
                    trips.append({
                        "trip_id": folder.name,
                        "partition": partition
                    })
    if not trips:
        raise HTTPException(status_code=404, detail="No trip data directories found under data/")
    return {"trips": trips}


@app.post("/api/predict/{trip_id}")
def predict_trip(trip_id: str, k: int = 16, beta: float = 15.0):
    """Process selected trip through the complete map-matching pipeline.
    
    1. Loads raw track and road network.
    2. Generates spatial candidates (k candidates per point).
    3. Runs ML model to get emission probabilities.
    4. Executes HMM Viterbi decoding for continuous sequence reconstruction.
    5. Calculates prediction accuracy using server-side reference data without leakage.
    6. Returns reconstructed route coordinates for Folium map visualization on Laptop 2.
    """
    t0 = time.time()
    
    # Locate trip directory
    trip_dir = None
    for partition in ["student_test", "train"]:
        candidate_dir = DATA_ROOT / partition / trip_id
        if candidate_dir.is_dir():
            trip_dir = candidate_dir
            break
            
    if trip_dir is None:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in dataset partitions.")
        
    try:
        # Load raw data
        d = load_trip(trip_dir)
        road_graph = RoadGraph.from_trip_dict(d)
        track = d["track"]
        nodes = d["nodes"]
        arcs = d["arcs"]
        route = d["route"]
        
        # Spatial candidate generation
        cands_df, cov = generate_candidates(trip_dir, k=k, include_reference=True)
        if len(cands_df) == 0:
            raise HTTPException(status_code=400, detail=f"Could not generate candidates for trip {trip_id}")
            
        # Model match scoring
        model = get_model()
        cands_df["pred_prob"] = model.predict_proba(cands_df)[:, 1]
        
        # HMM Viterbi route decoding
        decoder = HMMRouteDecoder(road_graph, beta=beta)
        decoded_results = decoder.decode(cands_df, prob_col="pred_prob")
        
        predicted_arc_ids = [res["chosen_arc_id"] for res in decoded_results]
        
        # Calculate server-side evaluation metrics (never exposed as raw labels to UI)
        correct_matches = sum(1 for res in decoded_results if res["label"] == 1)
        labeled_points = sum(1 for res in decoded_results if res["label"] in [0, 1])
        top1_acc = round(correct_matches / labeled_points * 100.0, 2) if labeled_points > 0 else 0.0
        
        continuity = round(evaluate_route_continuity(road_graph, predicted_arc_ids) * 100.0, 2)
        latency_ms = round((time.time() - t0) * 1000, 2)
        
        # Build coordinates for map visualization
        # Raw GPS: [lat, lon]
        gps_latlon = [[float(track[i, 1]), float(track[i, 0])] for i in range(len(track))]
        
        # Reconstructed route geometry: list of [[lat1, lon1], [lat2, lon2]] segments
        reconstructed_segments = []
        for aid in predicted_arc_ids:
            u, v = arcs[aid]
            reconstructed_segments.append([
                [float(nodes[u, 1]), float(nodes[u, 0])],
                [float(nodes[v, 1]), float(nodes[v, 0])]
            ])
            
        return {
            "trip_id": trip_id,
            "status": "success",
            "num_gps_observations": len(track),
            "num_reconstructed_arcs": len(predicted_arc_ids),
            "accuracy_percent": top1_acc,
            "route_continuity_percent": continuity,
            "latency_ms": latency_ms,
            "predicted_arc_ids": predicted_arc_ids[:50],  # sample preview
            "gps_coordinates": gps_latlon,
            "reconstructed_segments": reconstructed_segments
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
