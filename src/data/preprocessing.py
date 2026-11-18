"""Preprocessing helpers; raw source arrays remain unchanged."""
import numpy as np
EARTH_RADIUS_M=6371008.8

def project_lonlat(coords, origin=None):
    coords=np.asarray(coords,dtype=float)
    if coords.ndim!=2 or coords.shape[1]!=2 or len(coords)==0: raise ValueError("coords must be non-empty Nx2")
    lon0,lat0=np.nanmean(coords,axis=0) if origin is None else origin
    x=EARTH_RADIUS_M*np.radians(coords[:,0]-lon0)*np.cos(np.radians(lat0))
    y=EARTH_RADIUS_M*np.radians(coords[:,1]-lat0)
    return np.column_stack((x,y)),(float(lon0),float(lat0))

def prepare_track(track, sort_by_time=False):
    arr=np.asarray(track,dtype=float).copy()
    if arr.ndim!=2 or arr.shape[1]!=3: raise ValueError("track columns must be lon,lat,timestamp")
    original=np.arange(len(arr)); finite=np.isfinite(arr).all(axis=1)
    valid=finite&(np.abs(arr[:,0])<=180)&(np.abs(arr[:,1])<=90)
    if sort_by_time:
        order=np.argsort(arr[:,2],kind="stable"); arr=arr[order]; original=original[order]; valid=valid[order]
    dt=np.diff(arr[:,2])
    return {"track":arr,"original_observation_idx":original,"valid_mask":valid,
            "timestamp_deltas_s":dt,"nonincreasing_steps":int((dt<=0).sum()),
            "gaps_over_10s":int((dt>10).sum())}

def filter_gps_spikes(track_xy, timestamps, max_speed_mps=50.0):
    """Detect and flag GPS jumps where point-to-point speed exceeds realistic limits."""
    xy = np.asarray(track_xy, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    n = len(xy)
    if n <= 1:
        return np.ones(n, dtype=bool)
    
    keep = np.ones(n, dtype=bool)
    dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dts = np.diff(ts)
    speeds = np.zeros(n - 1, dtype=float)
    valid_dt = dts > 0
    speeds[valid_dt] = dists[valid_dt] / dts[valid_dt]
    
    # Flag points with unrealistic speeds (> max_speed_mps, e.g. 180 km/h)
    spike_idx = np.where(speeds > max_speed_mps)[0] + 1
    keep[spike_idx] = False
    return keep

def smooth_trajectory_xy(track_xy, alpha=0.3):
    """Exponential smoothing on Cartesian (x, y) coordinates to reduce high-frequency GPS jitter."""
    xy = np.asarray(track_xy, dtype=float)
    if len(xy) <= 2:
        return xy.copy()
    smoothed = np.empty_like(xy)
    smoothed[0] = xy[0]
    for i in range(1, len(xy)):
        smoothed[i] = alpha * xy[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed

def compute_kinematics(track_xy, timestamps):
    """Compute step distances, headings, speeds, and time deltas for projected GPS points."""
    xy = np.asarray(track_xy, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    n = len(xy)
    if n == 0:
        return {"distances_m": np.array([]), "speeds_mps": np.array([]),
                "headings_rad": np.array([]), "time_deltas_s": np.array([])}
    if n == 1:
        return {"distances_m": np.array([0.0]), "speeds_mps": np.array([0.0]),
                "headings_rad": np.array([0.0]), "time_deltas_s": np.array([0.0])}
    
    deltas = np.diff(xy, axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    dts = np.diff(ts)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    speeds = np.zeros(n - 1, dtype=float)
    pos_dt = dts > 0
    speeds[pos_dt] = dists[pos_dt] / dts[pos_dt]
    
    # Pad to match observation length n (forward fill last step)
    dists_padded = np.append(dists, dists[-1] if len(dists) else 0.0)
    speeds_padded = np.append(speeds, speeds[-1] if len(speeds) else 0.0)
    headings_padded = np.append(headings, headings[-1] if len(headings) else 0.0)
    dts_padded = np.append(dts, dts[-1] if len(dts) else 0.0)
    
    return {
        "distances_m": dists_padded,
        "speeds_mps": speeds_padded,
        "headings_rad": headings_padded,
        "time_deltas_s": dts_padded
    }

