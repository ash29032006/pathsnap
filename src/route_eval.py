"""Route-level evaluation: classifier-only vs classifier + HMM/Viterbi, per trip."""
import os

import numpy as np
import pandas as pd

from common import DATA, core


def evaluate_trip(part: str, trip_id: str, cand: pd.DataFrame, prob: np.ndarray, cfg: core.HMMConfig,
                  cache: dict | None = None) -> dict:
    key = (part, trip_id)
    if cache is not None and key in cache:
        trip, net = cache[key]
    else:
        trip = core.load_trip(os.path.join(DATA, part, trip_id), with_route=True)
        net = core.Network(trip)
        if cache is not None:
            cache[key] = (trip, net)
    n = len(trip.xy)
    feats = cand.drop(columns=["label_arc", "y", "trip"], errors="ignore")
    lab = cand.groupby("pt").label_arc.first()
    pts = lab.index.to_numpy(); lab = lab.to_numpy()

    co = core.classifier_only(feats, prob, n)
    hm, breaks = core.viterbi(trip, net, feats, prob, cfg)
    out = dict(trip=trip_id, n_points=n, hmm_breaks=breaks)
    for name, pred in (("clf", co), ("hmm", hm)):
        seq = core.compress(pred)
        full = core.assemble_route(net, seq)
        out[f"{name}_top1"] = float(np.mean(pred[pts] == lab))
        out[f"{name}_continuity"] = core.route_continuity(net, seq)
        out[f"{name}_continuity_filled"] = core.route_continuity(net, full)
        for k, v in core.route_set_metrics(full, trip.route, net).items():
            out[f"{name}_{k}"] = v
    return out


def evaluate_partition(part, df, prob, cfg, cache=None, trips=None):
    rows = []
    df = df.assign(_p=prob)
    for tid, g in df.groupby("trip"):
        if trips is not None and tid not in trips:
            continue
        rows.append(evaluate_trip(part, tid, g.drop(columns="_p"), g._p.to_numpy(), cfg, cache))
    r = pd.DataFrame(rows)
    return r


def summarize(r: pd.DataFrame) -> dict:
    """Point-weighted Top-1, mean of the per-trip route metrics."""
    w = r.n_points
    s = {}
    for c in r.columns:
        if c in ("trip", "n_points"):
            continue
        s[c] = float(np.average(r[c], weights=w)) if c.endswith("top1") else float(r[c].mean())
    return s
