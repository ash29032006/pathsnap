"""Evaluate the final pipeline on the evaluator_test folder and log it to MLflow.

1. Every trip goes through the running API (POST /predict), the same code path the UI uses,
   so the API terminal logs one PREDICT line per trip.
2. The same trips are also scored at candidate level (LightGBM probabilities vs aligned labels)
   to get observation-level Top-1, Macro F1 and log loss, as reported for student-test.
3. Folium maps (raw GPS, road network, reference route, predicted route) per trip.
4. MLflow experiment "4 - Evaluator test": one summary run + one child run per trip.

Outputs: results/evaluator_test/{per_trip.csv, summary.json, responses/*.json},
         results/maps/pred_evaluator_test_*.html
Usage:   python src/eval_evaluator.py --data ../evaluator_test --api http://localhost:8000
         (start the API first with PATHSNAP_EVAL_DIR pointing at the same folder)
"""
import argparse
import glob
import json
import logging
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import requests

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from common import ART, RES, ROOT, core, row_metrics, save_json  # noqa: E402
from decode import pred_map  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(RES, "evaluator_test")
EXP = "4 - Evaluator test"


def candidate_metrics(trip_dir, model):
    """Candidate-level scoring with labels from the reference route (same as build_dataset.py)."""
    trip = core.load_trip(trip_dir, with_route=True)
    net = core.Network(trip)
    cand = core.build_candidates(trip, net)
    lab = core.align_route(trip, net, trip.route)
    cand["y"] = (cand.arc == lab[cand.pt.to_numpy()]).astype(np.int8)
    cand["trip"] = trip.trip_id
    prob = model.predict_proba(cand[core.FEATURES].to_numpy(np.float32))[:, 1]
    return trip, net, cand, prob


def log_to_mlflow(per_trip, summary):
    import mlflow
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
    client = mlflow.MlflowClient()
    old = client.get_experiment_by_name(EXP)
    if old is not None:  # re-run: retire the previous experiment so the name can be reused
        client.rename_experiment(old.experiment_id, f"{EXP} (old {int(time.time())})")
        client.delete_experiment(old.experiment_id)
    exp_id = mlflow.create_experiment(
        EXP, artifact_location=f"file://{os.path.join(ROOT, 'mlruns')}",
        tags={"mlflow.note.content": f"Final pipeline (tuned LightGBM + HMM/Viterbi) on the "
              f"{len(per_trip)} trips of the evaluator_test folder, run through the API. "
              f"Source: src/eval_evaluator.py."})
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run(run_name="Evaluator test - final pipeline"):
        mlflow.set_tags({"stage": "evaluator_test", "selected_model": "true"})
        mlflow.log_params({"model": "LightGBM (tuned) + HMM/Viterbi", "n_trips": len(per_trip),
                           "data": summary["data"]})
        with open(os.path.join(ART, "hmm_config.json")) as f:
            h = json.load(f)
        mlflow.log_params({f"hmm_{k}": h[k] for k in ("beta", "emis_w", "break_logp", "max_speed")})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(os.path.join(OUT, "per_trip.csv"))
        mlflow.log_artifact(os.path.join(OUT, "summary.json"))
        for m in sorted(glob.glob(os.path.join(RES, "maps", "pred_evaluator_test_*.html"))):
            mlflow.log_artifact(m, "maps")
        for _, r in per_trip.iterrows():
            with mlflow.start_run(run_name=f"trip {r.trip}", nested=True):
                mlflow.set_tag("stage", "evaluator_trip")
                mlflow.log_param("trip", r.trip)
                mlflow.log_metrics({k: float(v) for k, v in r.items() if k != "trip"})
                mlflow.log_artifact(os.path.join(RES, "maps", f"pred_evaluator_test_{r.trip}.html"), "maps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "evaluator_test"))
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "responses"), exist_ok=True)

    api_trips = {t["trip_id"] for t in requests.get(a.api + "/trips", timeout=10).json()["trips"]}
    trips = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.data, "*")) if os.path.isdir(d))
    missing = [t for t in trips if t not in api_trips]
    if missing:
        raise SystemExit(f"API at {a.api} does not serve {missing}; start it with "
                         f"PATHSNAP_EVAL_DIR={os.path.abspath(a.data)}")
    model = joblib.load(os.path.join(ART, "models", "final_model.joblib"))

    rows, all_cand, all_prob = [], [], []
    for t in trips:
        r = requests.post(a.api + "/predict", json={"trip_id": t}, timeout=600)
        r.raise_for_status()
        resp = r.json()
        with open(os.path.join(OUT, "responses", f"{t}.json"), "w") as f:
            json.dump({k: v for k, v in resp.items() if k not in ("gps", "predicted_route")}, f, indent=2)
        trip, net, cand, prob = candidate_metrics(os.path.join(a.data, t), model)
        cm = row_metrics(cand, prob)
        all_cand.append(cand); all_prob.append(prob)
        pred_map(trip, net, resp["predicted_arcs"], f"Evaluator test {t}",
                 os.path.join(RES, "maps", f"pred_evaluator_test_{t}.html"))
        m = resp["metrics"]
        rows.append(dict(trip=t, n_points=resp["n_points"], n_candidates=resp["n_candidates"],
                         top1_hmm=m["top1_accuracy_hmm"], top1_classifier=m["top1_accuracy_classifier_only"],
                         continuity_hmm=m["route_continuity_hmm"],
                         continuity_classifier=m["route_continuity_classifier_only"],
                         route_precision=m["route_precision"], route_recall=m["route_recall"],
                         route_f1=m["route_f1"], hmm_breaks=m["hmm_breaks"],
                         points_with_candidates=m["points_with_candidates"],
                         cand_macro_f1=cm["macro_f1"], cand_log_loss=cm["log_loss"],
                         inference_ms=resp["timings"]["total_inference_ms"]))
        print(f"{t}  pts={resp['n_points']:5d}  top1 clf {m['top1_accuracy_classifier_only']:.4f} -> "
              f"HMM {m['top1_accuracy_hmm']:.4f}  route F1 {m['route_f1']:.4f}  "
              f"cont {m['route_continuity_hmm']:.4f}  {resp['timings']['total_inference_ms']:.0f} ms", flush=True)

    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(os.path.join(OUT, "per_trip.csv"), index=False)
    w = per_trip.n_points / per_trip.n_points.sum()
    pooled = row_metrics(pd.concat(all_cand, ignore_index=True), np.concatenate(all_prob))
    summary = dict(
        data=os.path.abspath(a.data), n_trips=len(per_trip), n_points=int(per_trip.n_points.sum()),
        top1_hmm=float((per_trip.top1_hmm * w).sum()),            # point-weighted, as in hmm_vs_classifier.csv
        top1_classifier=float((per_trip.top1_classifier * w).sum()),
        continuity_hmm=float(per_trip.continuity_hmm.mean()),
        continuity_classifier=float(per_trip.continuity_classifier.mean()),
        route_f1=float(per_trip.route_f1.mean()),
        route_precision=float(per_trip.route_precision.mean()),
        route_recall=float(per_trip.route_recall.mean()),
        cand_top1=pooled["top1"], cand_macro_f1=pooled["macro_f1"], cand_log_loss=pooled["log_loss"],
        mean_inference_ms=float(per_trip.inference_ms.mean()),
    )
    save_json(summary, os.path.join(OUT, "summary.json"))
    print("\n" + "=" * 70)
    for k, v in summary.items():
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    print("=" * 70)
    if not a.no_mlflow:
        log_to_mlflow(per_trip, summary)
        print(f'MLflow: experiment "{EXP}" logged')


if __name__ == "__main__":
    main()
