# PathSnap — Person 1 beginner setup (Windows + VS Code)

## What you are building

Your responsibility is the data pipeline: read each trip's four files, validate and preprocess GPS points, find nearby road arcs (candidate generation), calculate candidate features, and pass a documented table to Person 2.

## 1. Open the project

1. Extract this ZIP to a simple location, e.g. `C:\\Hackathon\\PathSnap_Person1`.
2. Open VS Code.
3. Select **File → Open Folder...** and choose `PathSnap_Person1`.
4. Open **Terminal → New Terminal**. The terminal should start in the project folder.

## 2. Confirm the data layout

The project must contain:

```text
PathSnap_Person1/
  data/
    train/
      00000001/
        00000001.track
        00000001.nodes
        00000001.arcs
        00000001.route
    student_test/
      ...
  src/
  requirements.txt
```

The dataset is included in this handoff. Do not rename the four file extensions or trip folders.

## 3. Create a Python environment (PowerShell terminal)

Run these commands one at a time:

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

When active, the terminal prompt normally starts with `(.venv)`.

## 4. Run the first dataset audit

From the project root:

```powershell
python src/eda.py --root data --out results
```

Expected broad result: 80 training trips and 10 student-test trips. The script writes `results/dataset_inventory.csv` and `results/eda_summary.json`.

## 5. Read one trip yourself

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from data.loader import load_trip; d=load_trip('data/train/00000001'); print(d['trip_id']); [print(k, v.shape, v[:2]) for k,v in d.items() if k in ('track','nodes','arcs','route')]"
```

Meaning of the files:
- `.track`: GPS observations, columns are longitude, latitude, timestamp.
- `.nodes`: road-network node coordinates, columns are longitude, latitude.
- `.arcs`: directed road links, each row contains start-node index and end-node index.
- `.route`: ordered reference arc IDs. An arc ID is the zero-based row number in `.arcs`.

## 6. Run a small candidate-generation smoke test

This is deliberately limited to 10 GPS observations per training trip so it finishes faster:

```powershell
python src/candidates/run_candidates.py --root data --partition train --k 64 --max-points 10 --out results
```

It creates a candidate CSV and a per-trip coverage CSV under `results/`.

Do not treat candidate miss rate as official map-matching accuracy. The current label is a geometric proxy: each GPS observation is associated with its nearest arc among the trip's reference-route arcs. The source data does not explicitly align each GPS observation to a route arc.

## 7. Git: create and push your own branch

First create an **empty** GitHub repository (do not add a README, licence, or .gitignore on GitHub). Copy its HTTPS URL.

In the VS Code terminal, from the project root, run:

```powershell
git --version
git init -b person1/data-pipeline
git status
git add .
git status
git commit -m "Set up Person 1 data pipeline"
git remote add origin PASTE_YOUR_GITHUB_REPOSITORY_HTTPS_URL_HERE
git push -u origin person1/data-pipeline
```

Replace the entire `PASTE...` text with your repository URL, e.g. `https://github.com/your-account/PathSnap.git`.

If Git asks who you are, set your commit identity (use your own details):

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

Then repeat the commit command. GitHub authentication should be completed through the browser/device prompt; never share your password or token in chat.

### Daily branch workflow

After editing code:

```powershell
git status
git add src\data src\candidates GETTING_STARTED_PERSON1.md
git commit -m "Improve candidate generation"
git push
```

To see the current branch:

```powershell
git branch --show-current
```

Keep working on `person1/data-pipeline`. Do not push directly to `main` unless your team lead explicitly asks. Once the team wants to integrate your work, open a Pull Request on GitHub from `person1/data-pipeline` into `main`.

## 8. What to send Person 2

Share the candidate table schema documented in `PERSON1_HANDOFF.md`. Candidate IDs are the original `.arcs` row indices. At inference time, do not read `.route`; student-test candidates are generated with labels `-1`.

## Important

The assistant can inspect and prepare the files attached in this chat workspace, but cannot take remote control of your physical laptop or click inside your local VS Code/GitHub session. Follow the commands here and paste any error output (not passwords/tokens) so we can troubleshoot together.
