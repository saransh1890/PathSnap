# PathSnap — Team Member Roles, Model Architecture & Two-Laptop Deployment Guide

**PES UNIVERSITY — Department of Computer Science and Engineering**  
**Course:** UE24CS352A: Machine Learning | Hackathon Project  
**Repository:** https://github.com/saransh1890/PathSnap

---

## Part 1: Exactly What Each Person Does in the Model & Pipeline

```text
                                [ RAW DATA ]
                         (90 Trips: .track, .nodes, .arcs)
                                      |
                                      v
   +-----------------------------------------------------------------------+
   |                     PERSON 1: SANKET SHIVAJI RATHOD                   |
   |                         (Data & Spatial Pipeline)                     |
   |  • GPS Preprocessing (Noise spike filter >50 m/s, metric projection)  |
   |  • Directed RoadGraph (88k nodes, 165k arcs, Dijkstra shortest path)  |
   |  • Candidate Generation (k=64 spatial nearest arcs via STRtree)       |
   |  • 19 Engineered Features (Distances, Emission proxy, Heading diff)   |
   |  • Baseline Classifiers: Logistic Regression & Distance-Weighted KNN  |
   +-----------------------------------+-----------------------------------+
                                       |
                   Candidate Probabilities & RoadGraph
                                       |
                                       +-------------------------+
                                       |                         |
                                       v                         v
   +-----------------------------------------+   +-----------------------------------------+
   |       PERSON 2: SANTHOSH M              |   |       PERSON 3: SARANSH MITTAL          |
   |  (Sequential Modeling & Reconstruction) |   |  (5-Model Benchmark & Optimization)     |
   |  • HMM State Representation (Arc States)|   |  • 5 Distinct Model Families:           |
   |  • Emission Probabilities P(obs|arc)    |   |    - Logistic Regression (Linear)       |
   |  • Transition Probabilities P(c_t|c_t-1)|   |    - K-Nearest Neighbours (Instance)    |
   |    via RoadGraph network distance       |   |    - Random Forest (Ensemble Bagging)   |
   |  • Viterbi Dynamic Programming Decoding |   |    - HistGradientBoosting (Boosting)    |
   |  • Route Continuity (19% -> 99.9% path) |   |    - Multi-Layer Perceptron (Neural Net)|
   +--------------------+--------------------+   |  • Bayesian Optimization (skopt GP)     |
                        |                        |  • Publication Charts (Accuracy, Loss)  |
                        v                        +--------------------+--------------------+
             Reconstructed Road Route                                 |
                        |                                             |
                        +----------------------+----------------------+
                                               |
                                               v
   +---------------------------------------------------------------------------------------+
   |                            PERSON 4: SANVIKA NAIK                                     |
   |                        (Two-Laptop System Deployment)                                 |
   |  • Laptop 1 (Model Host & API): FastAPI REST server (api/main.py)                     |
   |    - Endpoints: GET /health, GET /api/trips, POST /api/predict/{trip_id}               |
   |    - Server-side accuracy calculation with zero ground-truth leakage                   |
   |  • Laptop 2 (Evaluator UI): Streamlit Web Application (ui/app.py)                     |
   |    - Dropdown for 10 evaluation trips, Folium interactive maps, metric cards          |
   +---------------------------------------------------------------------------------------+
```

---

### Detailed Breakdown by Team Member

### 👤 Person 1: Sanket Shivaji Rathod (`PES1UG24AM251`)
**Assigned Deliverables:** GPS Data Preprocessing, Road-Network Graph Construction, Candidate Generation, Feature Engineering, Logistic Regression, KNN, and Evaluation Baseline.
- **Files Owned:** `src/data/preprocessing.py`, `src/data/run_preprocessing.py`, `src/graph/road_graph.py`, `src/candidates/candidate_generation.py`, `src/features/feature_engineering.py`, `src/models/train_logistic_regression.py`, `src/models/train_knn.py`.
- **What He Does in the Model:**
  1. **GPS Preprocessing:** Cleans 227,577 GPS points across 90 trips. Detects and removes 1,531 velocity spikes ($>50\text{ m/s}$ / $180\text{ km/h}$) caused by GPS multipath reflection. Applies exponential smoothing to suppress GPS jitter and projects spherical (lon, lat) to local Cartesian $(x, y)$ metric coordinates in metres.
  2. **Road-Network Graph:** Builds the directed `RoadGraph` from `.nodes` and directed `.arcs` (88,809 nodes, 165,790 road segments). Precomputes arc lengths, directed bearings, midpoints, topological adjacency (`successors`, `predecessors`), and fast Dijkstra shortest path routing.
  3. **Candidate Generation:** Uses Shapely `STRtree` spatial indexing with adaptive expanding buffer search to retrieve the top $k=64$ road arc candidates for each GPS observation.
  4. **Feature Engineering (19 Features):** Computes non-linear distances (`log_distance`, `inv_distance`, midpoint), Gaussian emission probability proxy $\exp\left(-\frac{1}{2}(d/\sigma)^2\right)$, circular heading alignment $|\theta_{\text{arc}} - \theta_{\text{gps}}|$, $\cos(\Delta\theta)$, speed-heading interaction, and competitive candidate rank within the observation.
  5. **Baseline Models:** Trains Logistic Regression ($C=0.01$, balanced class weights) and Distance-Weighted KNN ($k=21$) using 5-Fold `GroupKFold` cross-validation on trips to guarantee **zero data leakage**.

---

### 👤 Person 2: Santhosh M (`PES1UG24AM252`) — GitHub: [@Santhosh123-m](https://github.com/Santhosh123-m)
**Assigned Deliverables:** Hidden Markov Model (HMM), Viterbi Sequential Route Decoding, Transition & Emission Probability Modeling, Route Continuity Analysis.
- **Files Owned:** `src/hmm/route_reconstruction.py`, `src/hmm/__init__.py`.
- **What He Does in the Model:**
  1. **HMM State Formulation:** Defines the hidden states as candidate road arcs $c_t \in C_t$ for observation $t$.
  2. **Emission Modeling:** Calculates emission probabilities using the ML classifier's predicted match probability $P(\text{arc} \mid \text{obs})$ or Gaussian orthogonal distance:
     $$P(o_t \mid c_t) = \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left(-\frac{1}{2}\left(\frac{\text{dist}(o_t, c_t)}{\sigma_z}\right)^2\right)$$
  3. **Network Transition Modeling:** Calculates transition probability between consecutive candidate arcs based on the difference between observed GPS displacement and road network shortest path distance:
     $$P(c_t \mid c_{t-1}) = \frac{1}{\beta} \exp\left(-\frac{|\Delta d_{\text{euclid}} - \Delta d_{\text{network}}|}{\beta}\right)$$
     Calls Person 1's `RoadGraph.network_transition_distance()` to query Dijkstra road distances.
  4. **Viterbi Recurrence Algorithm:** Implements dynamic programming in log-space to find the globally optimal, physically connected sequence of road arcs across the entire trip:
     $$V_t(j) = \max_i \left[ V_{t-1}(i) + \ln P(c_t^j \mid c_{t-1}^i) \right] + \ln P(o_t \mid c_t^j)$$
  5. **Route Continuity:** Eliminates impossible vehicle teleportation and disconnected road jumps, boosting physical road connectivity from **~19% (raw classifier jumps) to up to 99.9% continuous road trajectory**.

---

### 👤 Person 3: Saransh Mittal (`PES1UG24AM254`)
**Assigned Deliverables:** 5 Model Families Benchmark, Bayesian Hyperparameter Optimization (`skopt`), Evaluation Plots & Visualizations.
- **Files Owned:** `src/models/train_all_models.py`, `src/optimization/bayesian_opt.py`, `src/visualization/generate_plots.py`, `results/plots/`.
- **What He Does in the Model:**
  1. **5 Distinct Model Families Benchmark:** Trains and evaluates models across 5 distinct algorithm families as required by the PES University hackathon guidelines:
     - **Linear / Probabilistic:** Logistic Regression (Top-1: 80.92%, ROC-AUC: 0.9899)
     - **Instance-based:** Distance-Weighted KNN (Top-1: 98.00%, ROC-AUC: 0.9948)
     - **Ensemble (Bagging):** Random Forest (Top-1: 94.50%, ROC-AUC: 0.9996)
     - **Boosting:** HistGradientBoosting (Top-1: 85.58%, ROC-AUC: 0.9969)
     - **Neural Network:** Multi-Layer Perceptron MLP (Top-1: 84.83%, ROC-AUC: 0.9929)
  2. **Bayesian Hyperparameter Optimization:** Uses `scikit-optimize` (`skopt.gp_minimize`) with a Gaussian Process surrogate to optimize `learning_rate`, `max_depth`, `max_iter`, `min_samples_leaf`, and `l2_regularization` with 3-Fold GroupKFold CV. Saves tuned artifact to `models/tuned_gradient_boosting.joblib`.
  3. **Visualizations:** Generates publication-quality charts for the final technical report:
     - `model_accuracy_comparison.png` (Top-1 vs Top-3 accuracy across all 5 models)
     - `cv_f1_and_auc_comparison.png` (Macro F1 and ROC-AUC)
     - `logloss_and_latency_comparison.png` (Log Loss calibration and inference latency)
     - `bayesian_optimization_trials.png` (Convergence curve across optimization trials)

---

### 👤 Person 4: Sanvika Naik (`PES1UG24AM253`)
**Assigned Deliverables:** Distributed System Architecture, Two-Laptop Deployment System, FastAPI Backend (`api/`), Streamlit Evaluator Interface (`ui/`).
- **Files Owned:** `api/main.py`, `ui/app.py`, `TWO_LAPTOP_SETUP.md`, `tests/test_api.py`.
- **What She Does in the Model:**
  1. **System Architecture & Repository Structure:** Created the base repository layout, directory hierarchy, and dependency environment.
  2. **Laptop 1 Backend (FastAPI REST Service):**
     - Exposes `GET /` (health status), `GET /api/trips` (lists the 10 evaluation trips), and `POST /api/predict/{trip_id}`.
     - Loads raw data, calls Person 1's candidate generator, calls ML model for emission scores, and calls Person 2's HMM Viterbi decoder.
     - Calculates server-side accuracy and route continuity without ever leaking ground-truth labels to Laptop 2.
  3. **Laptop 2 Frontend (Streamlit Evaluator UI):**
     - Creates the evaluator web dashboard running on Laptop 2.
     - Connects across the local network to Laptop 1's IP address.
     - Evaluator selects any of the 10 reserved trips from the dropdown and clicks **"Reconstruct Route & Evaluate"**.
     - Renders an interactive Folium map showing raw noisy GPS observations (blue dashed line) vs reconstructed road path (solid red line), start/end markers, and latency/accuracy metric badges.

---

## Part 2: Step-by-Step Instructions to Run on 2 Laptops

The hackathon guidelines mandate a **Two-Laptop Deployment**:
- **Laptop 1:** Runs the FastAPI Backend (hosts models, graph, HMM, server-side evaluation).
- **Laptop 2:** Runs the Streamlit Evaluator UI (displays evaluation trips, sends requests over LAN, shows Folium maps).

---

### Phase A: Network Setup (Connect Both Laptops)
1. Connect both **Laptop 1** and **Laptop 2** to the **same Wi-Fi network** (or connect Laptop 2 to Laptop 1's Mobile Hotspot).
2. On **Laptop 1**, open terminal and find its local IP address:
   ```powershell
   ipconfig
   ```
   Look for the **`IPv4 Address`** under your active Wi-Fi adapter (for example: `192.168.1.45`).  
   👉 *Note this IP address down!*

---

### Phase B: Start Laptop 1 (Model & API Host)

1. Open VS Code / Terminal on **Laptop 1**.
2. Navigate to the project root and activate the environment:
   ```powershell
   cd "PathSnap"
   .\.venv\Scripts\activate
   ```
3. Start the FastAPI backend server on all network interfaces (`0.0.0.0`):
   ```powershell
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. You should see:
   ```text
   INFO:     Started server process
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```
   ✅ *Laptop 1 is now running and waiting for requests from Laptop 2!*

---

### Phase C: Start Laptop 2 (Evaluator UI)

1. Open terminal on **Laptop 2**.
2. Navigate to the project root and activate the environment:
   ```powershell
   cd "PathSnap"
   .\.venv\Scripts\activate
   ```
3. Start the Streamlit application:
   ```powershell
   streamlit run ui/app.py
   ```
4. A browser window will open automatically displaying the **PathSnap Evaluator Dashboard**.
5. In the left sidebar:
   - In the **"FastAPI Backend URL"** input box, replace `http://localhost:8000` with Laptop 1's IP address:
     ```text
     http://192.168.1.45:8000
     ```
     *(Replace `192.168.1.45` with Laptop 1's actual IP)*
6. You will immediately see the green indicator:  
   👉 **`🟢 Connected to Laptop 1 API`**

---

### Phase D: Demonstrating to the Evaluator (The Live Demo)

1. Hand **Laptop 2** to the professor / evaluator.
2. Ask the evaluator to select any of the **10 reserved evaluation trips** from the dropdown menu (e.g., `00000000`, `00000007`, `00000009`, etc.).
3. Click the red button: **"🚀 Reconstruct Route & Evaluate"**.
4. Within 3-5 seconds:
   - **Laptop 1's terminal** will show the incoming request and inference logs.
   - **Laptop 2's screen** will show:
     - **Prediction Accuracy:** e.g. **94.97%**
     - **Route Continuity:** e.g. **99.92%**
     - **API Latency:** e.g. **4.9 seconds**
     - **Interactive Folium Map:** showing observed GPS points (blue dashed line) vs reconstructed physical road route (solid red line) across the city!
     - An expandable drawer with the complete reconstructed road arc sequence.

---

### 💡 Fallback / Single-Laptop Demo Mode:
If the evaluator asks you to demonstrate everything on a single laptop:
1. Open Terminal 1: `uvicorn api.main:app --port 8000`
2. Open Terminal 2: `streamlit run ui/app.py`
3. Leave the URL as `http://localhost:8000`. The entire system runs seamlessly!
