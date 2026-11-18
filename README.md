# PathSnap — Distributed Map-Matching & Route Reconstruction
**PES UNIVERSITY — Department of Computer Science and Engineering**  
**Course:** UE24CS352A: Machine Learning | Hackathon Project

---

## 👥 Team Members & Contribution Breakdown

| Role | Student Name | SRN | Assigned Topic & Core Contributions | Branch Name |
| :---: | :--- | :---: | :--- | :---: |
| **Person 1** | **Sanket Shivaji Rathod** | `PES1UG24AM251` | • **GPS Data Preprocessing:** Outlier spike filter ($>50\text{ m/s}$), smoothing, local metric Cartesian $(x, y)$ projection.<br>• **RoadGraph:** Directed road network (88k nodes, 165k arcs) with cached Dijkstra shortest-path routing.<br>• **Candidate Generation:** $k=64$ spatial candidate retrieval with Shapely `STRtree`.<br>• **Feature Engineering:** 19 spatial, angular, kinematic, and ranking features.<br>• **Baseline Models:** Logistic Regression ($C=0.01$) & Distance-Weighted KNN ($k=21$). | [`person1/data-pipeline`](https://github.com/saransh1890/PathSnap/tree/person1/data-pipeline) |
| **Person 2** | **Santhosh M** ([@Santhosh123-m](https://github.com/Santhosh123-m)) | `PES1UG24AM252` | • **HMM State Representation:** Candidate road arcs as hidden states with ML posterior emission modeling.<br>• **Network Transition Modeling:** $P(c_t \mid c_{t-1}) \propto \exp\left(-\frac{\|\Delta d_{\text{euclid}} - \Delta d_{\text{network}}\|}{\beta}\right)$ using `RoadGraph`.<br>• **Viterbi Sequence Decoding:** Global dynamic programming in log-space for continuous trajectory reconstruction.<br>• **Route Continuity Analysis:** Enforcing physical road connectivity (boosting continuity from ~19% to 99.9%). | [`person2/hmm-reconstruction`](https://github.com/saransh1890/PathSnap/tree/person2/hmm-reconstruction) |
| **Person 3** | **Saransh Mittal** ([@saransh1890](https://github.com/saransh1890)) | `PES1UG24AM254` | • **5 Model Families Benchmark:** Logistic Regression, KNN, Random Forest, HistGradientBoosting, and MLP Classifier.<br>• **Bayesian Hyperparameter Optimization:** Gaussian Process surrogate (`skopt.gp_minimize`) tuning `learning_rate`, `max_depth`, `l2_reg`.<br>• **Publication Visualizations:** Publication-ready evaluation plots (Accuracy, F1/AUC, Log Loss/Latency, and Bayesian trial curves). | [`person3/optimization-and-models`](https://github.com/saransh1890/PathSnap/tree/person3/optimization-and-models) |
| **Person 4** | **Sanvika Naik** | `PES1UG24AM253` | • **System Architecture & Scaffold:** Repository layout, integration branch maintenance, and dependency setup.<br>• **Laptop 1 (FastAPI Backend):** Distributed REST API (`api/main.py`) serving endpoints `/`, `/api/trips`, `/api/predict/{trip_id}` with zero ground-truth leakage.<br>• **Laptop 2 (Evaluator UI):** Streamlit application (`ui/app.py`) with Folium interactive maps, metric badges, and network controls. | [`person4/two-laptop-deployment`](https://github.com/saransh1890/PathSnap/tree/person4/two-laptop-deployment) |

---

## 📊 Consolidated 5-Model Evaluation Benchmark

Evaluated using 5-Fold `GroupKFold` cross-validation grouped by `trip_id` (zero data leakage):

| Model Name | Algorithm Family | Top-1 Accuracy | Top-3 Accuracy | CV Macro F1 | ROC-AUC | Log Loss | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | 80.92% | 95.46% | 0.7183 | 0.9899 | 0.0320 | 0.001 ms |
| **K-Nearest Neighbours** | Instance-based | **98.00%** | **98.46%** | **0.7936** | **0.9948** | **0.0116** | 0.015 ms |
| **Random Forest** | Ensemble (Bagging) | 94.50% | **99.62%** | **0.7940** | **0.9996** | 0.0377 | 0.002 ms |
| **HistGradientBoosting** | Boosting | 85.58% | 98.25% | **0.7984** | 0.9969 | 0.0222 | 0.001 ms |
| **Multi-Layer Perceptron (MLP)** | Neural Network | 84.83% | 96.83% | 0.7850 | 0.9929 | 0.0198 | 0.001 ms |

---

## 💻 Two-Laptop Distributed Deployment Architecture

```text
               [ Laptop 2 : Evaluator UI ]
                 Streamlit Web Application
                       (ui/app.py)
                           |
                           |  HTTP POST /api/predict/{trip_id}
                           v
               [ Laptop 1 : Model & API Host ]
                       FastAPI Server
                       (api/main.py)
                           |
         +-----------------+-----------------+
         |                 |                 |
         v                 v                 v
   [ ML Models ]     [ RoadGraph ]     [ HMM Viterbi ]
  (models/*.joblib)  (Dijkstra Paths)  (Sequence Decoder)
```

### Quick Run Instructions:

#### Laptop 1 (Model and API Server):
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Laptop 2 (Evaluator UI):
```powershell
streamlit run ui/app.py
```

---

## 🔬 Subsystem Execution Commands

- **Person 1 Live Demo:** `python src/run_pipeline_demo.py`
- **Person 2 HMM Decoding:** `python src/hmm/route_reconstruction.py --trip data/train/00000001`
- **Person 3 5-Model Benchmark:** `python src/models/train_all_models.py`
- **Person 3 Bayesian Optimization:** `python src/optimization/bayesian_opt.py --trials 10`
- **Person 3 Plot Generator:** `python src/visualization/generate_plots.py`
- **Person 4 API Integration Test:** `python tests/test_api.py`
