# PathSnap: map matching and route reconstruction

PathSnap takes noisy GPS observations from a vehicle trip and reconstructs the route it drove on the road network:

1. **Candidates.** For each GPS point, find the road arcs within 60 m (at most 10).
2. **Classifier.** Score each (point, candidate arc) pair: how likely is this the arc the vehicle was on?
3. **HMM / Viterbi.** Use road-network connectivity to choose a coherent sequence of arcs.
4. **Serving.** A FastAPI service (Laptop 1) and a browser UI (Laptop 2).

Results are in [RESULTS.md](RESULTS.md). Every number there comes from files in `results/`, which the scripts below regenerate.

---

## Repository layout

```
pathsnap/
├── src/
│   ├── core.py            # loading, projection, candidates, features, labels, HMM/Viterbi, route metrics
│   ├── build_dataset.py   # raw trips -> candidate feature tables + labels (data/features/)
│   ├── eda.py             # dataset exploration: inventory, quality checks, plots, Folium maps
│   ├── common.py          # shared paths + metric helpers
│   ├── train_models.py    # 8 models, 5-fold GroupKFold CV, student-test evaluation
│   ├── plots.py           # model comparison plots
│   ├── tune.py            # Bayesian optimisation (Optuna TPE) of LightGBM + OOF probabilities
│   ├── decode.py          # HMM parameter tuning, classifier vs HMM comparison, prediction maps
│   ├── route_eval.py      # per-trip route evaluation helpers
│   └── log_mlflow.py      # loads all results into MLflow for browsing
├── api/main.py            # FastAPI inference service
├── ui/                    # evaluator UI (static HTML/CSS/JS + Leaflet)
├── artifacts/             # final_model.joblib, hmm_config.json, model_config.json, all trained models
├── results/               # CSV results, Optuna trials, plots/, maps/, eda/
├── requirements.txt
└── RESULTS.md
```

## Setup

Python 3.10–3.12.

```bash
cd pathsnap
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If conda is installed, run `conda deactivate` before creating the venv. Otherwise the venv can end up linked to conda's Python and fail with `failed to parse CPython sys.version`. To fix it, delete `.venv` and recreate it from a python.org or Homebrew Python.

The data folder is expected at `../Student folder` (with `train/` and `student_test/` inside). To use a different location, set `PATHSNAP_DATA=/path/to/Student folder`.

## Reproduce the experiments

Run these in order from `pathsnap/` (times are from a laptop with 8+ cores):

```bash
python src/build_dataset.py      # ~15 s   candidate tables + labels for 90 trips
python src/eda.py                # ~1 min  results/eda/, results/maps/eda_*.html
python src/train_models.py       # ~10 min 8 models, CV + student-test, results/model_comparison.csv
python src/plots.py              #         results/plots/*.png
python src/tune.py --trials 40   # ~20 min Optuna search, artifacts/models/final_model.joblib
python src/decode.py --trials 25 # ~10 min HMM tuning, results/hmm_vs_classifier.csv, results/maps/pred_*.html
```

The Folium maps in `results/maps/` open in any browser.

## Browse the experiments in MLflow

`src/log_mlflow.py` loads every experiment above into a local MLflow store. It does not retrain anything; it reads the files in `results/` and `artifacts/`, so the numbers in MLflow are the same as in RESULTS.md.

```bash
python src/log_mlflow.py                                        # ~20 s, rebuilds mlflow.db + mlruns/
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002   # open http://localhost:5002
```

| Experiment | Runs |
|---|---|
| 1 - Model comparison | one run per classifier (8) with CV and student-test metrics, per-fold metrics, the MLP loss curve, confusion matrices and the trained model (all 8, about 540 MB in `mlruns/`) |
| 2 - Bayesian tuning (LightGBM) | the Optuna search with its 40 trials as child runs, plus LightGBM untuned vs tuned. The tuned model is registered as `PathSnap-LightGBM` |
| 3 - HMM decoding | the HMM parameter search with its 25 trials as child runs, plus nearest arc vs HMM distance-only vs classifier only vs classifier + HMM, and the Folium prediction maps |
| 4 - Evaluator test | the final pipeline on the 10 evaluator trips: one summary run with one child run per trip, plus the Folium maps (written by `src/eval_evaluator.py`) |

## Evaluate the evaluator_test folder

Start the API on that folder, then run every trip through it. The script calls `POST /predict` for each trip, so the API terminal logs every request. It also computes candidate-level Macro F1 and log loss, writes the Folium maps and logs experiment 4 to MLflow.

```bash
PATHSNAP_EVAL_DIR="../evaluator_test" uvicorn api.main:app --host 0.0.0.0 --port 8000   # terminal 1
python src/eval_evaluator.py --data ../evaluator_test --api http://localhost:8000      # terminal 2
```

Outputs: `results/evaluator_test/` (`per_trip.csv`, `summary.json`, `api_requests.log`, raw API responses) and `results/maps/pred_evaluator_test_*.html`.

## Deployment (two laptops)

Both laptops must be on the same network (the same Wi-Fi or a phone hotspot).

### Laptop 1: API

Put the 10 evaluator trips in one folder, one sub-folder per trip (`<id>/<id>.track|.nodes|.arcs|.route`), then run:

```bash
cd pathsnap && source .venv/bin/activate
export PATHSNAP_EVAL_DIR="/path/to/evaluator_test"     # default: ../Student folder/student_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find the laptop's IP address (macOS: `ipconfig getifaddr en0`, Windows: `ipconfig`). The terminal logs every request with the trip ID, accuracy and inference time. Show this log to the evaluator.

Check that it works from the same machine:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"trip_id":"00000056"}'
```

### Laptop 2: UI

Copy the `ui/` folder over (no Python packages are needed), then run:

```bash
cd pathsnap/ui
python3 -m http.server 5173
```

Open `http://localhost:5173/?api=http://<LAPTOP1_IP>:8000`, or type the API address into the page and press **Connect**. Select a trip and press **Run prediction**.

If it can't connect, allow port 8000 through Laptop 1's firewall (macOS asks the first time uvicorn runs; on Windows, allow Python on private networks).

### API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, model name, decoder parameters |
| GET | `/trips` | – | trip IDs in `PATHSNAP_EVAL_DIR` and their point counts |
| POST | `/predict` | `{"trip_id": "..."}` | predicted route (lat/lon polylines + arc IDs), sampled GPS, metrics, timings |

The prediction step loads only `.track`, `.nodes` and `.arcs`. The API reads `.route` after prediction, only to compute the metrics, and never returns it.

## Method summary

- **Projection.** Each trip uses a local equirectangular projection centred on its trajectory, giving metres. Trips are a few km across, so the projection error is far below GPS noise. Plain degree distances would be wrong because longitude shrinks with latitude, and the trips span many latitudes.
- **Labels.** The `.route` file lists the arcs in travel order but does not say which GPS point belongs to which arc. Each GPS point is aligned to a route position by monotonic dynamic programming: the position never goes backwards, and the total point-to-arc distance is minimised. That aligned arc is the point's label.
- **Features** (15 per candidate):
  - distance to the arc, its rank, the gap to the nearest candidate, and its ratio to the nearest distance
  - the angle between the GPS heading and the arc bearing, a stationary flag, and speed
  - arc length, and the position of the projection along the arc
  - the mean distance from the neighbouring ±3 and ±10 GPS points to the same arc
  - the heading difference over a wider ±5 window
  - the number of candidates, the out-degree at the arc's end node, and whether the arc is the reverse of the nearest one
- **Validation.** `GroupKFold(5)` grouped by trip, so no trip is split across folds. The student-test trips are used only for comparison.
- **HMM.**
  - Hidden states are the candidate arcs of each GPS point.
  - Emission is the classifier probability, renormalised over the point's candidates and raised to the power `emis_w`.
  - Transition is `exp(-|d_network − d_gps| / (beta·√Δt))`. `d_network` is the shortest driving distance along directed arcs between the two projected positions, found by a bounded Dijkstra search.
  - Unreachable pairs get a fixed low log-probability, so decoding continues across breaks.
  - Large time gaps widen the search radius and soften the transition.
  - Between consecutive decoded arcs that are not adjacent, shortest paths are inserted to produce a continuous route.
