"""Load one PathSnap trip without modifying source files."""
from pathlib import Path
import numpy as np

def _load(path, dtype=float):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    if path.stat().st_size == 0: return np.empty((0, 0), dtype=dtype)
    return np.loadtxt(path, dtype=dtype, ndmin=2)

def load_trip(trip_dir):
    trip_dir = Path(trip_dir); trip_id = trip_dir.name
    track = _load(trip_dir/f"{trip_id}.track", float)
    nodes = _load(trip_dir/f"{trip_id}.nodes", float)
    arcs = _load(trip_dir/f"{trip_id}.arcs", np.int64)
    route = _load(trip_dir/f"{trip_id}.route", np.int64).reshape(-1)
    if track.size and track.shape[1] != 3: raise ValueError(f"{trip_id}: track must have 3 columns")
    if nodes.size and nodes.shape[1] != 2: raise ValueError(f"{trip_id}: nodes must have 2 columns")
    if arcs.size and arcs.shape[1] != 2: raise ValueError(f"{trip_id}: arcs must have 2 columns")
    return {"trip_id":trip_id,"trip_dir":trip_dir,"track":track,"nodes":nodes,"arcs":arcs,"route":route}

def iter_trips(dataset_root, partition):
    folder=Path(dataset_root)/partition
    if not folder.is_dir(): raise FileNotFoundError(folder)
    for p in sorted(x for x in folder.iterdir() if x.is_dir()): yield load_trip(p)
