"""Shared helpers for training / evaluation scripts."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
import core  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "..", "Student folder"))
FEAT = os.path.join(ROOT, "data", "features")
RES = os.path.join(ROOT, "results")
ART = os.path.join(ROOT, "artifacts")
SEED = 42
for d in (RES, ART, os.path.join(RES, "plots"), os.path.join(RES, "maps"), os.path.join(ART, "models")):
    os.makedirs(d, exist_ok=True)


def load_partition(part: str, point_step: int = 1) -> pd.DataFrame:
    """Load cached candidate tables. point_step>1 keeps every k-th GPS point (all its candidates)."""
    dfs = []
    for f in sorted(glob.glob(os.path.join(FEAT, part, "*.pkl"))):
        d = pd.read_pickle(f)
        if point_step > 1:
            d = d[d.pt % point_step == 0]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def xy(df):
    return df[core.FEATURES].to_numpy(np.float32), df.y.to_numpy()


def top1(df: pd.DataFrame, prob: np.ndarray) -> float:
    """Observation-level Top-1: per GPS point, is the highest-scored candidate the reference arc?"""
    d = df[["trip", "pt", "y"]].assign(p=prob)
    idx = d.groupby(["trip", "pt"]).p.idxmax()
    return float(d.loc[idx, "y"].mean())


def row_metrics(df, prob, thr=0.5):
    y = df.y.to_numpy()
    yhat = (prob >= thr).astype(int)
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return dict(
        top1=top1(df, prob),
        macro_f1=f1_score(y, yhat, average="macro"),
        precision=precision_score(y, yhat, average="macro", zero_division=0),
        recall=recall_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, p, labels=[0, 1]),
    )


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
