# PathSnap — Two-Laptop Deployment Guide

According to the official PES University Hackathon specification:
- **Laptop 1 (Model and API):** Hosts the trained classifier, HMM/Viterbi implementation, road-network graph and server-side evaluation.
- **Laptop 2 (User Interface):** Displays the ten evaluation trips, sends the selected trip to Laptop 1, and displays the reconstructed route and evaluation metrics.

---

## 💻 Laptop 1 Setup (Model & FastAPI Host)

1. Connect Laptop 1 and Laptop 2 to the same Wi-Fi / hotspot network.
2. Find Laptop 1's local IP address:
   ```powershell
   ipconfig
   ```
   *(Look for `IPv4 Address`, e.g. `192.168.1.45`)*
3. Activate environment and start the FastAPI service:
   ```powershell
   cd PathSnap
   .\.venv\Scripts\activate
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Laptop 1 is now ready to receive inference calls from Laptop 2!

---

## 💻 Laptop 2 Setup (Evaluator UI)

1. Open a terminal on Laptop 2:
   ```powershell
   cd PathSnap
   .\.venv\Scripts\activate
   streamlit run ui/app.py
   ```
2. The Streamlit browser window will open automatically.
3. In the left sidebar:
   - If running on the same laptop for demo: enter `http://localhost:8000`
   - If running across two laptops: enter `http://<LAPTOP_1_IP>:8000` (e.g. `http://192.168.1.45:8000`)
4. Verify the connection indicator displays: **🟢 Connected to Laptop 1 API**.
5. Select any of the **10 evaluation trips** from the dropdown.
6. Click **"🚀 Reconstruct Route & Evaluate"**.
7. The UI will show:
   - Prediction Accuracy (%)
   - Route Continuity (%)
   - Latency in milliseconds
   - Interactive Folium map with raw GPS points vs HMM Reconstructed road route.

---

## 🔬 Running Individual Subsystems

### 1. Run 5-Model Benchmark Training
```powershell
python src/models/train_all_models.py
```
Trains Logistic Regression, KNN, Random Forest, HistGradientBoosting, and MLP Classifier. Saves `results/all_models_benchmark_summary.csv`.

### 2. Run Bayesian Hyperparameter Optimization
```powershell
python src/optimization/bayesian_opt.py --trials 10
```
Runs Gaussian Process Bayesian search and produces `results/bayesian_optimization_report.json` and `models/tuned_gradient_boosting.joblib`.

### 3. Run HMM Viterbi Route Reconstruction
```powershell
python src/hmm/route_reconstruction.py --trip data/train/00000001
```
Compares Classifier-only predictions vs Classifier + HMM sequence decoding.

### 4. Generate Publication Evaluation Charts
```powershell
python src/visualization/generate_plots.py
```
Creates PNG charts under `results/plots/` for the final technical report.
