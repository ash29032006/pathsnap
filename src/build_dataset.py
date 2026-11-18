"""Build candidate feature tables + labels for the train and student_test partitions.

Output: data/features/<partition>/<trip>.pkl  (one DataFrame per trip)
        results/candidate_stats.csv           (candidate miss rate per trip)

Usage: python src/build_dataset.py [--data "../Student folder"]
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import core  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def process(args):
    part, trip_dir, out_dir = args
    t0 = time.time()
    trip = core.load_trip(trip_dir, with_route=True)
    net = core.Network(trip)
    cand = core.build_candidates(trip, net)
    lab = core.align_route(trip, net, trip.route)
    cand["label_arc"] = lab[cand.pt.to_numpy()]
    cand["y"] = (cand.arc == cand.label_arc).astype(np.int8)
    cand["trip"] = trip.trip_id
    cand.to_pickle(os.path.join(out_dir, f"{trip.trip_id}.pkl"))
    n = len(trip.xy)
    has_cand = np.zeros(n, bool); has_cand[cand.pt.unique()] = True
    hit = np.zeros(n, bool); hit[cand.loc[cand.y == 1, "pt"].to_numpy()] = True
    return dict(partition=part, trip=trip.trip_id, n_points=n, n_rows=len(cand),
                no_candidate_rate=1 - has_cand.mean(), candidate_miss_rate=1 - hit.mean(),
                secs=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "Student folder"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    jobs = []
    for part in ("train", "student_test"):
        out = os.path.join(ROOT, "data", "features", part)
        os.makedirs(out, exist_ok=True)
        for d in sorted(glob.glob(os.path.join(a.data, part, "*"))):
            if os.path.isdir(d):
                jobs.append((part, d, out))
    with Pool(a.workers) as pool:
        rows = []
        for r in pool.imap_unordered(process, jobs):
            rows.append(r)
            print(f"{r['partition']:>12} {r['trip']} pts={r['n_points']:5d} rows={r['n_rows']:6d} "
                  f"miss={r['candidate_miss_rate']:.3f} {r['secs']}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["partition", "trip"])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "candidate_stats.csv"), index=False)
    print(df.groupby("partition")[["n_points", "n_rows", "no_candidate_rate", "candidate_miss_rate"]].agg(["sum", "mean"]))


if __name__ == "__main__":
    main()
