"""PathSnap inference API (runs on Laptop 1).

Endpoints
  GET  /health                 -> status, model info
  GET  /trips                  -> trip ids available in the evaluation folder (no labels)
  POST /predict  {"trip_id"}   -> runs the full pipeline on raw track/nodes/arcs, returns predicted
                                  route geometry + accuracy metrics computed server-side.

The reference route (.route) is read only AFTER prediction, only to score it, and is never
returned to the client. The prediction step loads the trip with with_route=False.

Config (environment variables)
  PATHSNAP_EVAL_DIR   folder containing one sub-folder per trip (default: ../Student folder/student_test)
  PATHSNAP_ARTIFACTS  folder with models/final_model.joblib + hmm_config.json (default: ../artifacts)
Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import core  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "..", "Student folder", "student_test"))
ART = os.environ.get("PATHSNAP_ARTIFACTS", os.path.join(ROOT, "artifacts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

MODEL = joblib.load(os.path.join(ART, "models", "final_model.joblib"))
with open(os.path.join(ART, "hmm_config.json")) as f:
    _h = json.load(f)
HMM = core.HMMConfig(beta=_h["beta"], emis_w=_h["emis_w"], break_logp=_h["break_logp"], max_speed=_h["max_speed"])
MODEL_CFG = json.load(open(os.path.join(ART, "model_config.json"))) if os.path.exists(
    os.path.join(ART, "model_config.json")) else {}

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _trip_dir(trip_id: str) -> str:
    if not trip_id.isalnum():
        raise HTTPException(400, "invalid trip id")
    d = os.path.join(EVAL_DIR, trip_id)
    if not os.path.isdir(d):
        raise HTTPException(404, f"trip {trip_id} not found")
    return d


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, 1000 * (time.time() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_CFG.get("model", "LightGBM"), "eval_dir": os.path.abspath(EVAL_DIR),
            "hmm": {"beta": HMM.beta, "emis_w": HMM.emis_w}}


@app.get("/trips")
def trips():
    out = []
    for t in sorted(os.listdir(EVAL_DIR)):
        d = os.path.join(EVAL_DIR, t)
        tr = os.path.join(d, f"{t}.track")
        if os.path.isdir(d) and os.path.exists(tr):
            with open(tr) as f:
                n = sum(1 for _ in f)
            out.append({"trip_id": t, "n_points": n})
    return {"trips": out, "count": len(out)}


def _latlon_segments(trip, arcs):
    segs, cur = [], []
    for e in arcs:
        f, t = trip.arcs[e]
        a = [float(trip.nodes[f, 1]), float(trip.nodes[f, 0])]
        b = [float(trip.nodes[t, 1]), float(trip.nodes[t, 0])]
        if cur and cur[-1] != a:
            segs.append(cur); cur = []
        if not cur:
            cur = [a]
        cur.append(b)
    if cur:
        segs.append(cur)
    return segs


@app.post("/predict")
def predict(req: PredictRequest):
    d = _trip_dir(req.trip_id)
    timings = {}
    t0 = time.time()
    # ---- prediction: raw track/nodes/arcs only, no reference route ----
    trip = core.load_trip(d, with_route=False)
    net = core.Network(trip)
    timings["load_ms"] = 1000 * (time.time() - t0); t1 = time.time()
    cand = core.build_candidates(trip, net)
    timings["candidates_ms"] = 1000 * (time.time() - t1); t1 = time.time()
    prob = MODEL.predict_proba(cand[core.FEATURES].to_numpy(np.float32))[:, 1] if len(cand) else np.array([])
    timings["classifier_ms"] = 1000 * (time.time() - t1); t1 = time.time()
    clf_pred = core.classifier_only(cand, prob, len(trip.xy))
    hmm_pred, breaks = core.viterbi(trip, net, cand, prob, HMM)
    seq = core.compress(hmm_pred)
    route = core.assemble_route(net, seq)
    timings["viterbi_ms"] = 1000 * (time.time() - t1)
    timings["total_inference_ms"] = 1000 * (time.time() - t0)

    # ---- server-side scoring against the reference route (never returned) ----
    t1 = time.time()
    ref = core.load_route(d)
    labels = core.align_route(trip, net, ref)
    has = hmm_pred >= 0
    metrics = {
        "top1_accuracy_hmm": float(np.mean(hmm_pred == labels)),
        "top1_accuracy_classifier_only": float(np.mean(clf_pred == labels)),
        "route_continuity_hmm": core.route_continuity(net, seq),
        "route_continuity_classifier_only": core.route_continuity(net, core.compress(clf_pred)),
        "points_with_candidates": float(has.mean()),
        "hmm_breaks": int(breaks),
        **core.route_set_metrics(route, ref, net),
    }
    timings["scoring_ms"] = 1000 * (time.time() - t1)

    step = max(1, len(trip.track) // 2000)
    resp = {
        "trip_id": req.trip_id,
        "n_points": int(len(trip.track)),
        "n_candidates": int(len(cand)),
        "predicted_arc_count": len(route),
        "predicted_route_km": float(net.length[route].sum() / 1000) if route else 0.0,
        "gps": trip.track[::step, [1, 0]].round(6).tolist(),
        "predicted_route": _latlon_segments(trip, route),
        "predicted_arcs": [int(x) for x in route],
        "metrics": metrics,
        "timings": {k: round(v, 1) for k, v in timings.items()},
    }
    log.info("PREDICT trip=%s points=%d top1_hmm=%.4f top1_clf=%.4f route_f1=%.4f continuity=%.4f infer=%.0fms",
             req.trip_id, resp["n_points"], metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier_only"],
             metrics["route_f1"], metrics["route_continuity_hmm"], timings["total_inference_ms"])
    return resp
