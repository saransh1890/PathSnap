"""Road-network graph representation and topological operations for PathSnap."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocessing import project_lonlat


class RoadGraph:
    """Directed road network graph constructed from nodes and directed arcs.
    
    Attributes:
        nodes: Nx2 float array of [lon, lat]
        arcs: Mx2 int array of [from_node, to_node]
        nodes_xy: Nx2 float array of projected metric coordinates [x, y]
        arc_lengths_m: M-dim float array of arc lengths in metres
        arc_bearings_rad: M-dim float array of directed headings in radians [-pi, pi]
        arc_midpoints_xy: Mx2 float array of arc midpoint coordinates
    """
    
    def __init__(self, nodes, arcs, origin=None):
        self.nodes = np.asarray(nodes, dtype=float)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.num_nodes = len(self.nodes)
        self.num_arcs = len(self.arcs)
        
        # Project nodes to local metric coordinates
        if self.num_nodes > 0:
            self.nodes_xy, self.origin = project_lonlat(self.nodes, origin=origin)
        else:
            self.nodes_xy = np.empty((0, 2), dtype=float)
            self.origin = origin or (0.0, 0.0)
            
        # Compute arc geometric properties
        if self.num_arcs > 0 and self.num_nodes > 0:
            self.starts_xy = self.nodes_xy[self.arcs[:, 0]]
            self.ends_xy = self.nodes_xy[self.arcs[:, 1]]
            vec = self.ends_xy - self.starts_xy
            self.arc_lengths_m = np.linalg.norm(vec, axis=1)
            self.arc_bearings_rad = np.arctan2(vec[:, 1], vec[:, 0])
            self.arc_midpoints_xy = (self.starts_xy + self.ends_xy) / 2.0
        else:
            self.starts_xy = np.empty((0, 2), dtype=float)
            self.ends_xy = np.empty((0, 2), dtype=float)
            self.arc_lengths_m = np.empty((0,), dtype=float)
            self.arc_bearings_rad = np.empty((0,), dtype=float)
            self.arc_midpoints_xy = np.empty((0, 2), dtype=float)
            
        # Build adjacency structures
        self._node_out_arcs = defaultdict(list)
        self._node_in_arcs = defaultdict(list)
        for arc_id, (u, v) in enumerate(self.arcs):
            self._node_out_arcs[int(u)].append(arc_id)
            self._node_in_arcs[int(v)].append(arc_id)
            
        # Cached spatial and matrix structures
        self._strtree = None
        self._lines = None
        self._adj_csr = None
        
    @classmethod
    def from_trip_dict(cls, trip_data):
        """Construct RoadGraph directly from load_trip dictionary."""
        return cls(trip_data["nodes"], trip_data["arcs"])
        
    @property
    def lines(self):
        """List of Shapely LineStrings for all arcs."""
        if self._lines is None:
            self._lines = [
                LineString((self.starts_xy[i], self.ends_xy[i]))
                for i in range(self.num_arcs)
            ]
        return self._lines

    @property
    def spatial_index(self):
        """Shapely STRtree spatial index for fast nearest-neighbor/radius queries."""
        if self._strtree is None and self.num_arcs > 0:
            self._strtree = STRtree(self.lines)
        return self._strtree

    @property
    def adjacency_matrix(self):
        """Weighted CSR adjacency matrix with edge weights = arc lengths in metres."""
        if self._adj_csr is None and self.num_arcs > 0:
            weights = np.maximum(self.arc_lengths_m, 1e-3)
            self._adj_csr = csr_matrix(
                (weights, (self.arcs[:, 0], self.arcs[:, 1])),
                shape=(self.num_nodes, self.num_nodes)
            )
        return self._adj_csr

    def outgoing_arcs(self, node_id):
        """Return list of arc IDs originating at node_id."""
        return self._node_out_arcs.get(int(node_id), [])

    def incoming_arcs(self, node_id):
        """Return list of arc IDs terminating at node_id."""
        return self._node_in_arcs.get(int(node_id), [])

    def arc_successors(self, arc_id):
        """Return list of arc IDs directly reachable from the end of arc_id."""
        to_node = self.arcs[arc_id, 1]
        return self.outgoing_arcs(to_node)

    def arc_predecessors(self, arc_id):
        """Return list of arc IDs directly leading into the start of arc_id."""
        from_node = self.arcs[arc_id, 0]
        return self.incoming_arcs(from_node)

    def are_arcs_connected(self, arc_u, arc_v):
        """Check if arc_v directly connects after arc_u (shares endpoint)."""
        return int(self.arcs[arc_u, 1]) == int(self.arcs[arc_v, 0])

    def shortest_path_distance(self, from_node, to_node):
        """Compute shortest network distance between two nodes in metres."""
        if from_node == to_node:
            return 0.0
        dist = dijkstra(
            self.adjacency_matrix,
            directed=True,
            indices=int(from_node),
            return_predecessors=False
        )
        return float(dist[int(to_node)])

    def network_transition_distance(self, arc_u, arc_v):
        """Network distance from end of arc_u to start of arc_v.
        
        Returns 0.0 if arc_v connects directly after arc_u.
        Returns shortest path distance if reachable.
        Returns np.inf if unreachable.
        """
        end_u = int(self.arcs[arc_u, 1])
        start_v = int(self.arcs[arc_v, 0])
        if end_u == start_v:
            return 0.0
        return self.shortest_path_distance(end_u, start_v)

    def connectivity_stats(self):
        """Compute connected component statistics for the road network."""
        if self.num_arcs == 0 or self.num_nodes == 0:
            return {"num_components": 0, "largest_component_nodes": 0}
        n_comp, labels = connected_components(self.adjacency_matrix, directed=False)
        largest = int(np.bincount(labels).max()) if len(labels) else 0
        return {"num_components": int(n_comp), "largest_component_nodes": largest}
