# Person 1 handoff — dataset and candidate pipeline

## Dataset format verified
Each trip directory contains four whitespace-delimited files:
- `.track`: longitude, latitude, timestamp (3 columns)
- `.nodes`: longitude, latitude (2 columns)
- `.arcs`: from-node index, to-node index (2 integer columns)
- `.route`: ordered arc IDs; each ID is the zero-based row index in `.arcs`

The 90 supplied trips are split into 80 `train` and 10 `student_test`. The evaluator partition was not supplied.

## Code
- `src/data/loader.py`: `load_trip(trip_dir)` returns raw NumPy arrays.
- `src/data/preprocessing.py`: local metre projection, GPS spike filtering, smoothing, and kinematics.
- `src/data/validation.py`: non-destructive validation checks.
- `src/graph/road_graph.py`: `RoadGraph` class with adjacency, successor/predecessor lookup, and shortest path.
- `src/candidates/candidate_generation.py`: `generate_candidates(trip_dir, k=64, ...)` returns `(DataFrame, coverage_dict)`.
- `src/candidates/run_candidates.py`: CLI for partition candidate export.
- `src/features/feature_engineering.py`: `CandidateFeaturePipeline` calculating 19 discriminative spatial, angular, and kinematic features.
- `src/models/train_logistic_regression.py`: GroupKFold CV training of Logistic Regression candidate ranker.
- `src/models/train_knn.py`: Distance-weighted KNN candidate ranker with hard negative sampling.
- `src/models/evaluate.py`: Benchmark comparing Baseline vs Logistic Regression vs KNN.
- `src/models/predict.py`: Inference pipeline generating candidate match probabilities for test trips.

## Candidate DataFrame contract
One row per observation-candidate pair:
`trip_id, observation_idx, arc_id, distance_m, midpoint_distance_m, segment_length_m, bearing_rad, projection_fraction, gps_heading_rad, gps_speed_mps, time_gap_s, reference_arc_id, label`

`arc_id` is the original zero-based row index in the `.arcs` file. Member 2 should join on `trip_id` + `observation_idx`; candidate labels are 1 for the provisional reference arc, 0 for other candidates, and -1 when reference labels are intentionally not loaded.

## Run commands (from repository root)
```bash
python src/eda.py --root data --out results
python src/candidates/run_candidates.py --root data --partition train --k 64 --max-points 30 --out results
python src/candidates/run_candidates.py --root data --partition train --k 64 --out results
python src/candidates/run_candidates.py --root data --partition student_test --k 64 --max-points 30 --out results
python src/visualize_trip.py --trip data/train/00000001 --out results/maps/train_00000001_reference.html

# Model training and evaluation
python src/models/train_logistic_regression.py
python src/models/train_knn.py
python src/models/evaluate.py
python src/models/predict.py --candidates results/student_test_candidate_sample_k64.csv --model models/logistic_regression.joblib --out results/student_test_predictions_lr.csv
```

## Benchmark Results (Person 1 Models)
Evaluated across 2,400 observations (80 trips):

| Method | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | MRR | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Geometric Nearest Baseline** | 57.96% | 93.25% | 95.79% | 0.7632 | 0.9742 | 0.5806 |
| **Logistic Regression** | 80.71% | 95.58% | 96.96% | 0.8848 | 0.9922 | 0.8084 |
| **K-Nearest Neighbours (KNN)** | 97.88% | 98.33% | 98.54% | 0.9851 | 0.9945 | 0.9804 |


## Important caveat about labels and coverage
The `.route` file gives an ordered route but does not provide a GPS-observation-to-arc alignment. For a supervised candidate label, this implementation assigns each GPS observation to the geometrically nearest arc among the trip's unique reference-route arcs. This is a transparent proxy, not verified ground-truth temporal alignment. Candidate miss rate therefore measures whether that proxy arc appears in the candidate set. Do not present it as exact official map-matching accuracy.

## Current measured development results
- All 90 supplied trips passed structural/range validation checks in the audit.
- 0 invalid GPS coordinate rows and 0 duplicate GPS rows were found in the inventory.
- Training has 590 adjacent timestamp gaps >10 seconds; largest is 153,064 seconds (trip `00000090`).
- Student test has 27 gaps >10 seconds; largest is 210 seconds.
- A deterministic 30-observation-per-trip sample with K=64 produced 152,303 train candidate rows over 2,400 observations and a weighted provisional candidate miss rate of 0.33%.
- Full-trajectory checks on difficult trips show K=64 miss rates of 5.34% for `00000033` and 10.32% for `00000054`; do not generalise the small-sample 0.33% figure to the full dataset.
- K=128 on 300 sampled observations gave 0% miss for trip `00000033` and 0.33% for trip `00000054`. Wider candidate sets improve recall but increase memory/training cost.

## Data leakage rules
- Never use `.route` labels to generate candidates at inference time.
- `include_reference=False` is used for student-test prediction inputs; labels are `-1`.
- Evaluator ground truth was not supplied and must not be fabricated.
