"""Create an interactive Folium EDA map for a supplied trip."""
from pathlib import Path
import argparse
import folium
from data.loader import load_trip

p=argparse.ArgumentParser(); p.add_argument('--trip',required=True); p.add_argument('--out',default='outputs/trip_map.html')
a=p.parse_args(); d=load_trip(a.trip); track,nodes,arcs,route=d['track'],d['nodes'],d['arcs'],d['route']
center=[float(track[:,1].mean()),float(track[:,0].mean())]
m=folium.Map(location=center,zoom_start=14,control_scale=True,tiles='OpenStreetMap')
folium.PolyLine([[float(y),float(x)] for x,y in track[:, :2]], color='blue', weight=3, opacity=.75, tooltip='Observed GPS trajectory').add_to(folium.FeatureGroup(name='GPS trajectory',show=True).add_to(m))
# Draw route geometry in supplied chronological arc order.
route_coords=[]
for aid in route:
    u,v=arcs[int(aid)]
    route_coords.append([float(nodes[u,1]),float(nodes[u,0])])
    route_coords.append([float(nodes[v,1]),float(nodes[v,0])])
if route_coords:
    folium.PolyLine(route_coords,color='green',weight=4,opacity=.8,tooltip='Supplied reference route').add_to(folium.FeatureGroup(name='Reference route',show=True).add_to(m))
folium.LayerControl(collapsed=False).add_to(m)
Path(a.out).parent.mkdir(parents=True,exist_ok=True); m.save(a.out)
print(f'Saved {a.out}; GPS points={len(track)}, route arcs={len(route)}')
