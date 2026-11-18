# PathSnap — distributed map matching and route reconstruction

This repository is being built from the supplied PES University PathSnap specification.

## Dataset layout

The supplied dataset is stored in `data/` and contains:

- `train/<trip_id>/<trip_id>.track|nodes|arcs|route`
- `student_test/<trip_id>/<trip_id>.track|nodes|arcs|route`

The evaluator partition is not included in the supplied files and must not be fabricated.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## First step: dataset audit

```bash
python src/eda.py --root data --out outputs
```

The script creates `dataset_inventory.csv` and `eda_summary.json`.

## Integrity

The `.route` files are reference labels. They are used only for training/development evaluation. Prediction-time functions must not read route labels. The evaluator labels must remain server-side and must never be passed to the model or UI.


## Person 1 — dataset/candidate pipeline
See `PERSON1_HANDOFF.md` for exact input/output contracts and commands.
Candidate generator:
```bash
python src/candidates/run_candidates.py --root data --partition train --k 64 --max-points 30 --out results
```
For full training candidates, omit `--max-points`; this can create a large file.


## Add the supplied dataset
Extract the organiser's `Student folder.zip`. It contains a top-level `Student folder/` directory. Copy the `train/` and `student_test/` folders from inside that directory into this project's `data/` folder so the final layout is:
- `data/train/<trip_id>/...`
- `data/student_test/<trip_id>/...`

Do not commit private/large organiser data to a public repository unless the organisers permit it.
