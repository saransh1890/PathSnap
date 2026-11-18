"""Spatial road-arc candidates. arc_id is the zero-based row in the .arcs file."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.loader import load_trip
from data.preprocessing import project_lonlat

def _segment_features(p,a,b):
    ab=b-a; den=float(np.dot(ab,ab))
    frac=0.0 if den<1e-12 else float(np.clip(np.dot(p-a,ab)/den,0,1))
    return float(np.linalg.norm(p-(a+frac*ab))),frac

def generate_candidates(trip_dir,k=12,search_radius_m=100.0,max_radius_m=2000.0,
                        include_reference=True,max_points=None):
    if k<1: raise ValueError("k must be >= 1")
    d=load_trip(trip_dir); track,nodes,arcs,route=(d[x] for x in ("track","nodes","arcs","route"))
    if not len(track) or not len(nodes) or not len(arcs):
        return pd.DataFrame(),{"trip_id":d["trip_id"],"points_used":0,"candidate_misses":0,"candidate_miss_rate":0.0}
    gps_xy,origin=project_lonlat(track[:,:2]); nodes_xy,_=project_lonlat(nodes,origin)
    starts,ends=nodes_xy[arcs[:,0]],nodes_xy[arcs[:,1]]
    vec=ends-starts; lengths=np.linalg.norm(vec,axis=1); mids=(starts+ends)/2
    bearings=np.arctan2(vec[:,1],vec[:,0]); lines=[LineString((starts[i],ends[i])) for i in range(len(arcs))]
    tree=STRtree(lines)
    obs=np.arange(len(track)) if max_points is None or len(track)<=max_points else np.unique(np.linspace(0,len(track)-1,max_points).round().astype(int))
    route_ids=np.unique(route) if include_reference and len(route) else np.array([],dtype=int)
    route_tree=STRtree([lines[int(i)] for i in route_ids]) if len(route_ids) else None
    rows=[]; misses=0
    for oi in obs:
        pxy=gps_xy[oi]; point=Point(float(pxy[0]),float(pxy[1])); radius=float(search_radius_m)
        ids=np.asarray(tree.query(point.buffer(radius)),dtype=int)
        while len(ids)<min(k,len(arcs)) and radius<max_radius_m:
            radius=min(radius*2,max_radius_m); ids=np.asarray(tree.query(point.buffer(radius)),dtype=int)
        if not len(ids): ids=np.array([int(tree.nearest(point))])
        dists=np.array([point.distance(lines[int(i)]) for i in ids])
        ids=ids[np.argsort(dists,kind="stable")[:k]]
        ref=-1
        if route_tree is not None: ref=int(route_ids[int(route_tree.nearest(point))])
        if ref>=0 and ref not in ids: misses+=1
        pi=max(0,oi-1); ni=min(len(track)-1,oi+1); dt=float(track[ni,2]-track[pi,2])
        delta=gps_xy[ni]-gps_xy[pi]; heading=float(np.arctan2(delta[1],delta[0])) if np.linalg.norm(delta)>0 else 0.0
        speed=float(np.linalg.norm(delta)/dt) if dt>0 else 0.0
        gap=float(track[oi,2]-track[oi-1,2]) if oi>0 else 0.0
        for aid in ids:
            dist,frac=_segment_features(pxy,starts[aid],ends[aid])
            rows.append({"trip_id":d["trip_id"],"observation_idx":int(oi),"arc_id":int(aid),
                "distance_m":dist,"midpoint_distance_m":float(np.linalg.norm(mids[aid]-pxy)),
                "segment_length_m":float(lengths[aid]),"bearing_rad":float(bearings[aid]),
                "projection_fraction":frac,"gps_heading_rad":heading,"gps_speed_mps":speed,
                "time_gap_s":max(gap,0.0),"reference_arc_id":ref,
                "label":int(aid==ref) if ref>=0 else -1})
    n=len(obs); miss_rate=(misses/n) if n and route_tree is not None else None
    return pd.DataFrame(rows),{"trip_id":d["trip_id"],"points_used":n,"points_total":len(track),
        "candidate_misses":misses if route_tree is not None else None,"candidate_miss_rate":miss_rate}

def build_partition(dataset_root,partition="train",k=12,max_points=None):
    root=Path(dataset_root)/partition; frames=[]; coverage=[]
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        df,cov=generate_candidates(folder,k=k,include_reference=(partition=="train"),max_points=max_points)
        if len(df): frames.append(df)
        coverage.append(cov)
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(),pd.DataFrame(coverage))
