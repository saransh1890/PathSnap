"""Dataset inventory and data-quality audit for PathSnap."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from data.loader import load_trip


def inspect_trip(folder, partition):
    d = load_trip(folder)
    t, n, a, r = d['track'], d['nodes'], d['arcs'], d['route']
    dt = np.diff(t[:, 2])
    valid_gps = np.isfinite(t[:, :2]).all(axis=1) & (np.abs(t[:, 0]) <= 180) & (np.abs(t[:, 1]) <= 90)
    n_nodes = len(n)
    graph = csr_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(graph, directed=False, return_labels=True)
    degrees = np.bincount(a.ravel(), minlength=n_nodes)
    return {
        'partition': partition, 'trip_id': d['trip_id'],
        'track_rows': len(t), 'nodes': len(n), 'arcs': len(a), 'route_arcs': len(r),
        'track_bytes': (folder / f"{d['trip_id']}.track").stat().st_size,
        'nodes_bytes': (folder / f"{d['trip_id']}.nodes").stat().st_size,
        'arcs_bytes': (folder / f"{d['trip_id']}.arcs").stat().st_size,
        'route_bytes': (folder / f"{d['trip_id']}.route").stat().st_size,
        'missing_or_invalid_gps': int((~valid_gps).sum()),
        'timestamp_nonincreasing': int((dt <= 0).sum()),
        'timestamp_gap_gt_10s': int((dt > 10).sum()),
        'timestamp_gap_max_s': float(dt.max()) if len(dt) else 0.0,
        'timestamp_step_median_s': float(np.median(dt)) if len(dt) else 0.0,
        'lon_min': float(np.nanmin(t[:,0])), 'lon_max': float(np.nanmax(t[:,0])),
        'lat_min': float(np.nanmin(t[:,1])), 'lat_max': float(np.nanmax(t[:,1])),
        'isolated_nodes': int((degrees == 0).sum()),
        'weak_components': int(n_comp),
        'largest_component_nodes': int(np.bincount(labels).max()) if len(labels) else 0,
        'route_arc_ids_valid': bool(len(r) == 0 or (r.min() >= 0 and r.max() < len(a))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Root containing train/ and student_test/')
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for partition in ('train','student_test'):
        for folder in sorted((root/partition).iterdir()):
            if folder.is_dir(): rows.append(inspect_trip(folder, partition))
    df=pd.DataFrame(rows)
    df.to_csv(out/'dataset_inventory.csv', index=False)
    summary={}
    for p,g in df.groupby('partition'):
        summary[p]={
            'trips': int(len(g)), 'gps_observations_total': int(g.track_rows.sum()),
            'gps_observations_mean': float(g.track_rows.mean()),
            'nodes_mean': float(g.nodes.mean()), 'arcs_mean': float(g.arcs.mean()),
            'route_arcs_mean': float(g.route_arcs.mean()),
            'invalid_gps_total': int(g.missing_or_invalid_gps.sum()),
            'nonincreasing_timestamp_steps': int(g.timestamp_nonincreasing.sum()),
            'gaps_over_10s': int(g.timestamp_gap_gt_10s.sum()),
            'max_timestamp_gap_s': float(g.timestamp_gap_max_s.max()),
            'weak_components_max': int(g.weak_components.max()),
        }
    (out/'eda_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('\nLargest timestamp gaps:')
    print(df.nlargest(10,'timestamp_gap_max_s')[['partition','trip_id','track_rows','timestamp_gap_max_s','timestamp_gap_gt_10s']].to_string(index=False))

if __name__=='__main__': main()
