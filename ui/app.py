"""Streamlit evaluator web application for PathSnap (Laptop 2)."""
import time
import requests
import streamlit as st
import folium
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PathSnap — Distributed Map Matching",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap — Distributed Map Matching & Route Reconstruction")
st.markdown("**Two-Laptop Deployment System — Evaluator Interface (Laptop 2)**")

# Sidebar: API connection & trip selection
st.sidebar.header("🔌 Laptop 1 Connection (API Host)")
api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://localhost:8000")

# Check connectivity
conn_status = st.sidebar.empty()
try:
    health_resp = requests.get(f"{api_url.rstrip('/')}/", timeout=2)
    if health_resp.status_code == 200:
        conn_status.success("🟢 Connected to Laptop 1 API")
    else:
        conn_status.warning(f"⚠️ API returned status {health_resp.status_code}")
except Exception:
    conn_status.error("🔴 Disconnected from Laptop 1. Ensure API is running on Laptop 1.")

# Fetch available evaluation trips
st.sidebar.header("🚗 Select Evaluation Trip")
trips_list = ["00000000", "00000007", "00000009", "00000040", "00000052", 
              "00000056", "00000065", "00000076", "00000077", "00000091",
              "00000001", "00000002"]

try:
    trips_resp = requests.get(f"{api_url.rstrip('/')}/api/trips", timeout=3)
    if trips_resp.status_code == 200:
        fetched = [t["trip_id"] for t in trips_resp.json().get("trips", [])]
        if fetched:
            trips_list = fetched
except Exception:
    pass

selected_trip = st.sidebar.selectbox("Trip ID", trips_list, index=0)
k_cands = st.sidebar.slider("Candidate Search Radius (k)", min_value=8, max_value=32, value=16, step=4)
beta_param = st.sidebar.slider("HMM Transition Parameter (β)", min_value=5.0, max_value=30.0, value=15.0, step=2.5)

run_button = st.sidebar.button("🚀 Reconstruct Route & Evaluate", type="primary")

# Main Interface
if run_button:
    with st.spinner(f"Sending Trip {selected_trip} to Laptop 1 for Map Matching & Route Reconstruction..."):
        try:
            req_payload = {"k": k_cands, "beta": beta_param}
            endpoint = f"{api_url.rstrip('/')}/api/predict/{selected_trip}?k={k_cands}&beta={beta_param}"
            
            t_start = time.time()
            resp = requests.post(endpoint, timeout=30)
            latency = round((time.time() - t_start) * 1000, 1)
            
            if resp.status_code != 200:
                st.error(f"Error from API ({resp.status_code}): {resp.text}")
            else:
                data = resp.json()
                
                # Top metrics banner
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("GPS Observations", f"{data['num_gps_observations']}")
                col2.metric("Prediction Accuracy", f"{data['accuracy_percent']}%")
                col3.metric("Route Continuity", f"{data['route_continuity_percent']}%")
                col4.metric("API Latency", f"{data['latency_ms']} ms")
                
                # Folium Map
                st.subheader(f"📍 Interactive Geographical Trajectory Map — Trip {selected_trip}")
                
                gps_coords = data.get("gps_coordinates", [])
                reconstructed_segs = data.get("reconstructed_segments", [])
                
                if gps_coords:
                    # Calculate center
                    center_lat = sum(p[0] for p in gps_coords) / len(gps_coords)
                    center_lon = sum(p[1] for p in gps_coords) / len(gps_coords)
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
                    
                    # 1. Raw GPS Trajectory (Blue dashed polyline)
                    fg_gps = folium.FeatureGroup(name="Raw Noisy GPS Points", show=True)
                    folium.PolyLine(
                        gps_coords,
                        color="blue",
                        weight=3,
                        opacity=0.6,
                        dash_array="5, 10",
                        tooltip="Observed GPS Trajectory"
                    ).add_to(fg_gps)
                    
                    # Add start and end pins
                    folium.Marker(gps_coords[0], popup="Trip Start", icon=folium.Icon(color="green", icon="play")).add_to(fg_gps)
                    folium.Marker(gps_coords[-1], popup="Trip End", icon=folium.Icon(color="black", icon="stop")).add_to(fg_gps)
                    fg_gps.add_to(m)
                    
                    # 2. Reconstructed Road Route (Red/Green solid polyline)
                    if reconstructed_segs:
                        fg_route = folium.FeatureGroup(name="HMM Reconstructed Route", show=True)
                        for seg in reconstructed_segs:
                            folium.PolyLine(
                                seg,
                                color="red",
                                weight=5,
                                opacity=0.85,
                                tooltip="Matched Road Arc"
                            ).add_to(fg_route)
                        fg_route.add_to(m)
                        
                    folium.LayerControl(collapsed=False).add_to(m)
                    
                    # Render map
                    components.html(m._repr_html_(), height=550)
                    
                # Reconstructed route arc preview table
                with st.expander("📋 View Reconstructed Road Arc Sequence"):
                    st.write(f"Total reconstructed arcs: {len(data['predicted_arc_ids'])}")
                    st.write(data["predicted_arc_ids"])
                    
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to Laptop 1 at `{api_url}`. Please make sure the FastAPI server is running on Laptop 1 with: `uvicorn api.main:app --host 0.0.0.0 --port 8000`")
        except Exception as ex:
            st.error(f"An unexpected error occurred: {ex}")
else:
    st.info("👈 Select an evaluation trip from the left sidebar and click **'Reconstruct Route & Evaluate'** to run the prediction pipeline over the network.")
