"""Non-destructive validation checks."""
import numpy as np

def validate_trip(d):
    t,n,a,r=(d[k] for k in ("track","nodes","arcs","route")); issues=[]
    if t.size and (t.ndim!=2 or t.shape[1]!=3): issues.append("invalid_track_shape")
    if n.size and (n.ndim!=2 or n.shape[1]!=2): issues.append("invalid_nodes_shape")
    if a.size and (a.ndim!=2 or a.shape[1]!=2): issues.append("invalid_arcs_shape")
    if t.size:
        if not np.isfinite(t).all(): issues.append("nonfinite_track_values")
        if ((np.abs(t[:,0])>180)|(np.abs(t[:,1])>90)).any(): issues.append("invalid_gps_coordinates")
        if (np.diff(t[:,2])<=0).any(): issues.append("nonincreasing_timestamps")
    if n.size:
        if not np.isfinite(n).all() or ((np.abs(n[:,0])>180)|(np.abs(n[:,1])>90)).any(): issues.append("invalid_nodes")
    if a.size and (a.min()<0 or a.max()>=len(n)): issues.append("arc_endpoint_out_of_range")
    if r.size and (r.min()<0 or r.max()>=len(a)): issues.append("route_arc_id_out_of_range")
    return {"trip_id":d["trip_id"],"valid":not issues,"issues":issues,
            "num_track":len(t),"num_nodes":len(n),"num_arcs":len(a),"num_route":len(r)}
