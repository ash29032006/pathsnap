"""PathSnap core: data loading, projection, candidate generation, features, labels, HMM/Viterbi.

Pipeline per trip:
  raw files -> local metric projection -> candidate road arcs per GPS point
  -> per-(point, candidate) feature rows -> classifier probability
  -> HMM/Viterbi over candidates using road-network connectivity -> route.
"""
from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R = 6371008.8

# Candidate generation settings (chosen from training-partition exploration, see eda.py)
CAND_RADIUS = 60.0      # metres
CAND_MAX_K = 10         # max candidates per GPS point
DENSIFY_STEP = 10.0     # metres between densified points on arcs for the KD-tree

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "heading_diff", "stationary", "speed",
    "arc_len", "frac",
    "ctx3_dist", "ctx10_dist", "ctx3_head",
    "n_cand", "out_deg", "is_rev_of_nearest",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t
    nodes: np.ndarray            # (V,2) lon, lat
    arcs: np.ndarray             # (E,2) from, to
    route: np.ndarray | None = None  # reference arc ids (kept server/label side only)
    # derived
    xy: np.ndarray = field(default=None, repr=False)        # projected GPS (N,2)
    nxy: np.ndarray = field(default=None, repr=False)       # projected nodes (V,2)
    origin: tuple = (0.0, 0.0)


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = lambda ext: os.path.join(trip_dir, f"{tid}.{ext}")
    track = np.loadtxt(p("track"), ndmin=2)
    nodes = np.loadtxt(p("nodes"), ndmin=2)
    arcs = np.loadtxt(p("arcs"), dtype=np.int64, ndmin=2)
    route = None
    if with_route and os.path.exists(p("route")):
        route = np.loadtxt(p("route"), dtype=np.int64, ndmin=1)
    trip = Trip(tid, track, nodes, arcs, route)
    clean_track(trip)
    project(trip)
    return trip


def load_route(trip_dir: str) -> np.ndarray:
    tid = os.path.basename(os.path.normpath(trip_dir))
    return np.loadtxt(os.path.join(trip_dir, f"{tid}.route"), dtype=np.int64, ndmin=1)


def clean_track(trip: Trip) -> None:
    """Drop non-finite / out-of-range rows, sort by time, drop duplicate timestamps."""
    tr = trip.track
    ok = np.isfinite(tr).all(1) & (np.abs(tr[:, 0]) <= 180) & (np.abs(tr[:, 1]) <= 90)
    tr = tr[ok]
    tr = tr[np.argsort(tr[:, 2], kind="stable")]
    keep = np.r_[True, np.diff(tr[:, 2]) > 1e-6]
    trip.track = tr[keep]


def project(trip: Trip) -> None:
    """Local equirectangular projection (metres) centred on the trajectory.

    Trips span only a few km, so the distortion of this projection is far below GPS noise.
    """
    lon0, lat0 = trip.track[:, 0].mean(), trip.track[:, 1].mean()
    c = np.cos(np.radians(lat0))
    f = lambda ll: np.c_[np.radians(ll[:, 0] - lon0) * c * EARTH_R, np.radians(ll[:, 1] - lat0) * EARTH_R]
    trip.origin = (lon0, lat0)
    trip.xy = f(trip.track[:, :2])
    trip.nxy = f(trip.nodes)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def point_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points p to segments a->b (all (...,2)). Returns dist, frac in [0,1]."""
    ab = b - a
    L2 = (ab ** 2).sum(-1)
    t = np.where(L2 > 0, ((p - a) * ab).sum(-1) / np.maximum(L2, 1e-12), 0.0)
    t = np.clip(t, 0, 1)
    proj = a + t[..., None] * ab
    return np.sqrt(((p - proj) ** 2).sum(-1)), t


def ang_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


class Network:
    """Road graph for one trip in projected coordinates."""

    def __init__(self, trip: Trip):
        self.a = trip.nxy[trip.arcs[:, 0]]
        self.b = trip.nxy[trip.arcs[:, 1]]
        self.frm = trip.arcs[:, 0]
        self.to = trip.arcs[:, 1]
        self.length = np.linalg.norm(self.b - self.a, axis=1)
        d = self.b - self.a
        self.bearing = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360) % 360  # 0 = north
        nV = len(trip.nxy)
        self.out_deg = np.bincount(self.frm, minlength=nV)
        # adjacency: node -> list of (arc_id)
        order = np.argsort(self.frm, kind="stable")
        self._adj_arcs = order
        self._adj_ptr = np.r_[0, np.cumsum(np.bincount(self.frm, minlength=nV))]
        # reverse-arc lookup
        key = {(int(f), int(t)): i for i, (f, t) in enumerate(trip.arcs)}
        self.rev = np.array([key.get((int(t), int(f)), -1) for f, t in trip.arcs])
        self._sp_cache: dict[int, tuple[float, dict]] = {}
        self._build_index()

    def _build_index(self):
        n = np.maximum(1, np.ceil(self.length / DENSIFY_STEP).astype(int)) + 1
        ids = np.repeat(np.arange(len(n)), n)
        ts = np.concatenate([np.linspace(0, 1, k) for k in n])
        pts = self.a[ids] + ts[:, None] * (self.b[ids] - self.a[ids])
        self.tree = cKDTree(pts)
        self.tree_ids = ids

    def out_arcs(self, node: int):
        return self._adj_arcs[self._adj_ptr[node]:self._adj_ptr[node + 1]]

    def dist_from(self, node: int, cutoff: float) -> dict:
        """Bounded Dijkstra over nodes from `node`; cached (re-run if larger cutoff needed)."""
        c = self._sp_cache.get(node)
        if c is not None and c[0] >= cutoff:
            return c[1]
        dist = {node: 0.0}
        pq = [(0.0, node)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for e in self.out_arcs(u):
                v = int(self.to[e]); nd = d + self.length[e]
                if nd < dist.get(v, np.inf) and nd <= cutoff:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._sp_cache[node] = (cutoff, dist)
        return dist

    def shortest_arc_path(self, src_arc: int, dst_arc: int, cutoff: float = 3000.0):
        """Arc sequence connecting end of src_arc to start of dst_arc (exclusive), or None."""
        s, g = int(self.to[src_arc]), int(self.frm[dst_arc])
        if s == g:
            return []
        dist = {s: 0.0}; prev = {}
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == g:
                break
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for e in self.out_arcs(u):
                v = int(self.to[e]); nd = d + self.length[e]
                if nd < dist.get(v, np.inf):
                    dist[v] = nd; prev[v] = int(e)
                    heapq.heappush(pq, (nd, v))
        if g not in prev:
            return None
        path = []; u = g
        while u != s:
            e = prev[u]; path.append(e); u = int(self.frm[e])
        return path[::-1]


# ---------------------------------------------------------------------------
# Candidates + features (no labels used here -> safe for inference)
# ---------------------------------------------------------------------------
def gps_heading_speed(xy: np.ndarray, t: np.ndarray, k: int = 2):
    n = len(xy)
    i0 = np.clip(np.arange(n) - k, 0, n - 1)
    i1 = np.clip(np.arange(n) + k, 0, n - 1)
    d = xy[i1] - xy[i0]
    dist = np.linalg.norm(d, axis=1)
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    head = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360) % 360
    return head, dist / dt, dist


def build_candidates(trip: Trip, net: Network, radius: float = CAND_RADIUS, k: int = CAND_MAX_K) -> pd.DataFrame:
    xy, t = trip.xy, trip.track[:, 2]
    hits = net.tree.query_ball_point(xy, r=radius + DENSIFY_STEP)
    pi, ai = [], []
    for i, h in enumerate(hits):
        if h:
            u = np.unique(net.tree_ids[h])
            pi.append(np.full(len(u), i)); ai.append(u)
    if not pi:
        return pd.DataFrame(columns=["pt", "arc"] + FEATURES)
    pi = np.concatenate(pi); ai = np.concatenate(ai)
    dist, frac = point_seg(xy[pi], net.a[ai], net.b[ai])
    df = pd.DataFrame({"pt": pi, "arc": ai, "dist": dist, "frac": frac})
    df = df[df.dist <= radius].sort_values(["pt", "dist"], kind="stable")
    df["dist_rank"] = df.groupby("pt").cumcount()
    df = df[df.dist_rank < k].reset_index(drop=True)

    pi = df.pt.to_numpy(); ai = df.arc.to_numpy()
    g = df.groupby("pt").dist
    mn = g.transform("min").to_numpy()
    df["dist_minus_min"] = df.dist - mn
    df["dist_ratio"] = df.dist / (mn + 1.0)
    df["n_cand"] = g.transform("size").to_numpy()

    head, speed, moved = gps_heading_speed(xy, t, 2)
    stationary = moved < 3.0
    hd = ang_diff(head[pi], net.bearing[ai])
    df["heading_diff"] = np.where(stationary[pi], 90.0, hd)
    df["stationary"] = stationary[pi].astype(float)
    df["speed"] = speed[pi]
    df["arc_len"] = net.length[ai]
    df["out_deg"] = net.out_deg[net.to[ai]]

    n = len(xy)
    for K, name in ((3, "ctx3_dist"), (10, "ctx10_dist")):
        acc = np.zeros(len(df))
        offs = [o for o in range(-K, K + 1) if o != 0]
        for o in offs:
            j = np.clip(pi + o, 0, n - 1)
            acc += point_seg(xy[j], net.a[ai], net.b[ai])[0]
        df[name] = acc / len(offs)
    head3, _, moved3 = gps_heading_speed(xy, t, 5)
    df["ctx3_head"] = np.where(moved3[pi] < 5.0, 90.0, ang_diff(head3[pi], net.bearing[ai]))

    nearest = df[df.dist_rank == 0].set_index("pt").arc
    near_arc = nearest.reindex(pi).to_numpy()
    df["is_rev_of_nearest"] = (net.rev[near_arc.astype(int)] == ai).astype(float)
    return df[["pt", "arc"] + FEATURES]


# ---------------------------------------------------------------------------
# Labels (training / server-side evaluation only)
# ---------------------------------------------------------------------------
def align_route(trip: Trip, net: Network, route: np.ndarray) -> np.ndarray:
    """Assign each GPS point a reference arc by monotonic DP alignment along the route.

    The .route file lists arcs in travel order but gives no per-point labels; we find the
    non-decreasing mapping point -> route position that minimises total point-to-arc distance.
    """
    xy = trip.xy
    A, B = net.a[route], net.b[route]
    N, M = len(xy), len(route)
    back = np.zeros((N, M), dtype=np.int32)
    ar = np.arange(M)
    prev = None
    for i in range(N):
        c = point_seg(xy[i][None, :], A, B)[0]
        if prev is None:
            prev = c; back[i] = ar; continue
        cm = np.minimum.accumulate(prev)
        # index of running minimum
        is_new = np.r_[True, prev[1:] < cm[:-1]]
        idx = np.maximum.accumulate(np.where(is_new, ar, 0))
        prev = c + cm; back[i] = idx
    j = int(np.argmin(prev)); out = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        out[i] = route[j]; j = back[i, j]
    return out


# ---------------------------------------------------------------------------
# HMM / Viterbi
# ---------------------------------------------------------------------------
@dataclass
class HMMConfig:
    beta: float = 5.0          # transition scale (m): penalty on |route_dist - gps_dist|
    emis_w: float = 1.0        # weight on classifier log-probability
    break_logp: float = -30.0  # log-prob for transitions with no network path (HMM break)
    max_speed: float = 50.0    # m/s, bounds Dijkstra search


def viterbi(trip: Trip, net: Network, cand: pd.DataFrame, prob: np.ndarray, cfg: HMMConfig):
    """Hidden states = candidate arcs of each GPS point (with projection fraction).

    Emission  : log of classifier probability renormalised over the point's candidates.
    Transition: -|d_network - d_gps| / beta (Newson & Krumm style), where d_network is
                the driving distance along directed arcs between the two projected positions.
    Returns per-point predicted arc (-1 where the point has no candidates) and break count.
    """
    xy, t = trip.xy, trip.track[:, 2]
    cand = cand.assign(p=prob)
    groups = {pt: g for pt, g in cand.groupby("pt", sort=True)}
    pts = sorted(groups)
    pred = np.full(len(xy), -1, dtype=np.int64)
    if not pts:
        return pred, 0
    states, scores, backs = [], [], []
    breaks = 0
    prev_pt = None
    for pt in pts:
        g = groups[pt]
        arcs = g.arc.to_numpy(); fr = g.frac.to_numpy()
        p = g.p.to_numpy(); p = p / max(p.sum(), 1e-12)
        em = cfg.emis_w * np.log(np.clip(p, 1e-9, 1))
        if prev_pt is None:
            sc = em; bk = np.full(len(arcs), -1)
        else:
            pa, pf = states[-1]
            dg = np.linalg.norm(xy[pt] - xy[prev_pt])
            dt = max(t[pt] - t[prev_pt], 1.0)
            cutoff = max(200.0, cfg.max_speed * dt + 100.0)
            beta = cfg.beta * np.sqrt(dt)
            T = np.full((len(pa), len(arcs)), cfg.break_logp)
            for i, (a, f) in enumerate(zip(pa, pf)):
                rem = (1 - f) * net.length[a]
                dmap = None
                for j, (b, fb) in enumerate(zip(arcs, fr)):
                    if a == b:
                        dr = abs(fb - f) * net.length[a] * (1.0 if fb >= f else 1.5)
                    else:
                        if dmap is None:
                            dmap = net.dist_from(int(net.to[a]), cutoff)
                        d0 = dmap.get(int(net.frm[b]))
                        if d0 is None:
                            continue
                        dr = rem + d0 + fb * net.length[b]
                    T[i, j] = -abs(dr - dg) / beta
            tot = scores[-1][:, None] + T
            bk = tot.argmax(0)
            sc = tot[bk, np.arange(len(arcs))] + em
            if (T.max() <= cfg.break_logp):
                breaks += 1
        states.append((arcs, fr)); scores.append(sc); backs.append(bk)
        prev_pt = pt
    j = int(np.argmax(scores[-1]))
    for k in range(len(pts) - 1, -1, -1):
        pred[pts[k]] = states[k][0][j]
        j = backs[k][j] if backs[k][j] >= 0 else j
    return pred, breaks


def classifier_only(cand: pd.DataFrame, prob: np.ndarray, n_points: int) -> np.ndarray:
    pred = np.full(n_points, -1, dtype=np.int64)
    if len(cand):
        c = cand.assign(p=prob)
        best = c.loc[c.groupby("pt").p.idxmax()]
        pred[best.pt.to_numpy()] = best.arc.to_numpy()
    return pred


# ---------------------------------------------------------------------------
# Route assembly + metrics
# ---------------------------------------------------------------------------
def compress(seq):
    out = []
    for a in seq:
        if a >= 0 and (not out or out[-1] != a):
            out.append(int(a))
    return out


def route_continuity(net: Network, arcs: list) -> float:
    """Fraction of consecutive distinct predicted arcs that are directly connected."""
    if len(arcs) < 2:
        return 1.0
    a = np.array(arcs)
    return float(np.mean(net.to[a[:-1]] == net.frm[a[1:]]))


def assemble_route(net: Network, arcs: list) -> list:
    """Fill gaps between non-adjacent consecutive arcs with shortest network paths."""
    if not arcs:
        return []
    out = [arcs[0]]
    for b in arcs[1:]:
        a = out[-1]
        if net.to[a] != net.frm[b]:
            path = net.shortest_arc_path(a, b)
            if path:
                out.extend(path)
        out.append(b)
    return out


def route_set_metrics(pred_route: list, ref_route: np.ndarray, net: Network) -> dict:
    """Length-weighted precision/recall/F1 between predicted and reference arc sets."""
    P, R = set(pred_route), set(int(x) for x in ref_route)
    L = lambda s: float(sum(net.length[x] for x in s))
    inter = L(P & R)
    prec = inter / max(L(P), 1e-9); rec = inter / max(L(R), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
