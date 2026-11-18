"""Dataset exploration for the train and student_test partitions.

Outputs (results/eda/):
  inventory.csv          per-trip file presence, sizes, row counts, time gaps, extent, network stats
  eda_summary.json       partition-level summary
  plots/*.png            distributions of trip length, sampling interval, GPS-to-road distance, etc.
  maps/eda_<trip>.html   Folium map: raw GPS, nearby road network, reference route
Usage: python src/eda.py
"""
import glob
import os

import folium
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DATA, RES, core, save_json  # noqa: E402

OUT = os.path.join(RES, "eda")
os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
os.makedirs(os.path.join(RES, "maps"), exist_ok=True)


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * core.EARTH_R * np.arcsin(np.sqrt(a))


def inspect(part, d):
    tid = os.path.basename(d)
    files = {ext: os.path.join(d, f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}
    row = dict(partition=part, trip=tid)
    for ext, f in files.items():
        row[f"has_{ext}"] = os.path.exists(f)
        row[f"{ext}_kb"] = round(os.path.getsize(f) / 1024, 1) if os.path.exists(f) else np.nan
    raw = pd.read_csv(files["track"], sep=r"\s+", header=None, names=["lon", "lat", "t"])
    row["track_rows"] = len(raw)
    row["track_missing"] = int(raw.isna().sum().sum())
    row["track_invalid_coord"] = int(((raw.lon.abs() > 180) | (raw.lat.abs() > 90)).sum())
    dt = np.diff(raw.t.to_numpy())
    row["non_increasing_t"] = int((dt <= 0).sum())
    row["dt_median"] = float(np.median(dt)); row["dt_max"] = float(dt.max())
    row["pct_dt_1s"] = float(np.mean(np.isclose(dt, 1.0)))
    row["gaps_over_5s"] = int((dt > 5).sum())
    row["duration_s"] = float(raw.t.iloc[-1] - raw.t.iloc[0])
    step = haversine(raw.lon.values[:-1], raw.lat.values[:-1], raw.lon.values[1:], raw.lat.values[1:])
    row["length_km"] = float(step.sum() / 1000)
    row["max_jump_m"] = float(step.max())
    speed = step / np.maximum(dt, 1e-6)
    row["max_speed_mps"] = float(speed.max())
    row["lon_min"], row["lon_max"] = raw.lon.min(), raw.lon.max()
    row["lat_min"], row["lat_max"] = raw.lat.min(), raw.lat.max()
    row["extent_km"] = float(haversine(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"]) / 1000)

    trip = core.load_trip(d)
    net = core.Network(trip)
    V, E = len(trip.nodes), len(trip.arcs)
    row.update(nodes=V, arcs=E, route_len=len(trip.route),
               route_unique=len(set(trip.route)),
               route_continuity=float(np.mean(net.to[trip.route[:-1]] == net.frm[trip.route[1:]])))
    g = coo_matrix((np.ones(E), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(V, V))
    nw, lab_w = connected_components(g, directed=True, connection="weak")
    ns, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=V)
    row.update(weak_components=nw, strong_components=ns,
               largest_weak_frac=float(np.bincount(lab_w).max() / V),
               largest_strong_frac=float(np.bincount(lab_s).max() / V),
               isolated_nodes=int((deg == 0).sum()),
               two_way_frac=float(np.mean(net.rev >= 0)),
               arc_len_median_m=float(np.median(net.length)))
    # GPS-to-nearest-arc distance (projected metres)
    dist, _ = net.tree.query(trip.xy)
    row["gps_to_road_median_m"] = float(np.median(dist))
    row["gps_to_road_p95_m"] = float(np.percentile(dist, 95))
    row["route_length_km"] = float(net.length[trip.route].sum() / 1000)
    return row, dist, dt


def folium_map(part, tid, path):
    trip = core.load_trip(os.path.join(DATA, part, tid))
    net = core.Network(trip)
    lo, la = trip.track[:, 0], trip.track[:, 1]
    m = folium.Map(location=[la.mean(), lo.mean()], zoom_start=14, tiles="OpenStreetMap")
    # nearby network only (arcs with a node within 150 m of the track bbox)
    x0, y0 = trip.xy.min(0) - 150; x1, y1 = trip.xy.max(0) + 150
    near = ((net.a[:, 0] > x0) & (net.a[:, 0] < x1) & (net.a[:, 1] > y0) & (net.a[:, 1] < y1))
    fg = folium.FeatureGroup("Road network (nearby)")
    for e in np.where(near)[0][:6000]:
        f, t = trip.arcs[e]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color="#9aa0a6", weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)
    fr = folium.FeatureGroup("Reference route")
    pts = [[trip.nodes[trip.arcs[trip.route[0], 0], 1], trip.nodes[trip.arcs[trip.route[0], 0], 0]]]
    pts += [[trip.nodes[trip.arcs[e, 1], 1], trip.nodes[trip.arcs[e, 1], 0]] for e in trip.route]
    folium.PolyLine(pts, color="#1f6b45", weight=5, opacity=0.8, tooltip="Reference route").add_to(fr)
    fr.add_to(m)
    fg2 = folium.FeatureGroup("Raw GPS")
    for i in range(0, len(lo), max(1, len(lo) // 1500)):
        folium.CircleMarker([la[i], lo[i]], radius=2, color="#2b59c3", fill=True, fill_opacity=0.8,
                            weight=0).add_to(fg2)
    fg2.add_to(m)
    folium.Marker([la[0], lo[0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([la[-1], lo[-1]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[la.min(), lo.min()], [la.max(), lo.max()]])
    m.save(path)


def main():
    rows, dists, dts = [], {"train": [], "student_test": []}, {"train": [], "student_test": []}
    for part in ("train", "student_test"):
        for d in sorted(glob.glob(os.path.join(DATA, part, "*"))):
            r, dist, dt = inspect(part, d)
            rows.append(r); dists[part].append(dist); dts[part].append(dt)
            print(part, r["trip"], r["track_rows"], r["nodes"], r["arcs"], flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)

    summ = {}
    for part, g in inv.groupby("partition"):
        summ[part] = dict(
            trips=len(g), total_gps_points=int(g.track_rows.sum()),
            gps_points_per_trip=g.track_rows.describe().round(1).to_dict(),
            missing_values=int(g.track_missing.sum()), invalid_coords=int(g.track_invalid_coord.sum()),
            non_increasing_timestamps=int(g.non_increasing_t.sum()),
            pct_dt_exactly_1s=float(np.mean(np.concatenate([np.isclose(x, 1) for x in dts[part]]))),
            trips_with_gaps_over_5s=int((g.gaps_over_5s > 0).sum()), max_gap_s=float(g.dt_max.max()),
            nodes_per_trip=g.nodes.describe().round(1).to_dict(), arcs_per_trip=g.arcs.describe().round(1).to_dict(),
            trips_with_isolated_nodes=int((g.isolated_nodes > 0).sum()),
            mean_weak_components=float(g.weak_components.mean()),
            mean_two_way_frac=float(g.two_way_frac.mean()),
            reference_routes_fully_connected=int((g.route_continuity == 1).sum()),
            trips_with_repeated_route_arcs=int((g.route_unique < g.route_len).sum()),
            gps_to_road_median_m=float(np.median(np.concatenate(dists[part]))),
            gps_to_road_p95_m=float(np.percentile(np.concatenate(dists[part]), 95)),
            gps_to_road_p99_m=float(np.percentile(np.concatenate(dists[part]), 99)),
        )
    anomalies = inv[(inv.dt_max > 10) | (inv.max_speed_mps > 60) | (inv.gps_to_road_p95_m > 30)]
    anomalies[["partition", "trip", "dt_max", "max_speed_mps", "gps_to_road_p95_m", "max_jump_m"]] \
        .to_csv(os.path.join(OUT, "anomalies.csv"), index=False)
    summ["anomalous_trips"] = anomalies.trip.tolist()
    save_json(summ, os.path.join(OUT, "eda_summary.json"))

    P = os.path.join(OUT, "plots")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, c in (("train", "#5b3fa8"), ("student_test", "#1f6b45")):
        g = inv[inv.partition == part]
        ax[0].hist(g.track_rows, bins=25, alpha=0.6, color=c, label=part)
        ax[1].hist(g.length_km, bins=25, alpha=0.6, color=c, label=part)
        ax[2].hist(g.arcs, bins=25, alpha=0.6, color=c, label=part)
    for a_, t in zip(ax, ["GPS points per trip", "Trip length (km)", "Arcs in trip network"]):
        a_.set_title(t); a_.legend()
    plt.tight_layout(); plt.savefig(os.path.join(P, "trip_distributions.png"), dpi=120); plt.close()

    alldt = np.concatenate(dts["train"])
    plt.figure(figsize=(6, 4))
    vals, cnts = np.unique(np.round(alldt), return_counts=True)
    plt.bar([str(int(v)) if v < 6 else ">5" for v in vals[:7]], cnts[:7], color="#5b3fa8")
    plt.yscale("log"); plt.title("Sampling interval (s), train"); plt.xlabel("seconds"); plt.ylabel("count (log)")
    plt.tight_layout(); plt.savefig(os.path.join(P, "sampling_interval.png"), dpi=120); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(np.clip(np.concatenate(dists["train"]), 0, 60), bins=60, color="#5b3fa8")
    plt.axvline(core.CAND_RADIUS, color="#b42318", ls="--", label=f"candidate radius {core.CAND_RADIUS:.0f} m")
    plt.title("GPS point to nearest road (m), train"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(P, "gps_to_road_distance.png"), dpi=120); plt.close()

    plt.figure(figsize=(9, 4.5))
    for part, c in (("train", "#5b3fa8"), ("student_test", "#1f6b45")):
        g = inv[inv.partition == part]
        plt.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, s=25, c=c, label=part)
    plt.xlabel("longitude"); plt.ylabel("latitude"); plt.title("Trip locations (centre of each trajectory)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(P, "trip_locations.png"), dpi=120); plt.close()

    # one trajectory over its network, projected metres
    trip = core.load_trip(os.path.join(DATA, "train", "00000034")); net = core.Network(trip)
    plt.figure(figsize=(7, 7))
    x0, y0 = trip.xy.min(0) - 200; x1, y1 = trip.xy.max(0) + 200
    near = (net.a[:, 0] > x0) & (net.a[:, 0] < x1) & (net.a[:, 1] > y0) & (net.a[:, 1] < y1)
    for e in np.where(near)[0]:
        plt.plot([net.a[e, 0], net.b[e, 0]], [net.a[e, 1], net.b[e, 1]], c="#c9ccc4", lw=0.6)
    for e in trip.route:
        plt.plot([net.a[e, 0], net.b[e, 0]], [net.a[e, 1], net.b[e, 1]], c="#1f6b45", lw=2.5)
    plt.scatter(trip.xy[:, 0], trip.xy[:, 1], s=1.5, c="#2b59c3")
    plt.axis("equal"); plt.title("Trip 00000034: GPS (blue), reference route (green), network (grey)")
    plt.tight_layout(); plt.savefig(os.path.join(P, "trajectory_00000034.png"), dpi=130); plt.close()

    for part, tid in (("train", "00000034"), ("train", "00000098"), ("student_test", "00000056")):
        folium_map(part, tid, os.path.join(RES, "maps", f"eda_{part}_{tid}.html"))
    print()
    for part in ("train", "student_test"):
        s = summ[part]
        print(f"=== {part}")
        print(f"  trips: {s['trips']}  |  total GPS points: {s['total_gps_points']:,}")
        print(f"  missing values: {s['missing_values']}  |  invalid coords: {s['invalid_coords']}")
        print(f"  sampling ~1s: {s['pct_dt_exactly_1s']*100:.1f}%  |  max gap: {s['max_gap_s']:.0f}s  |  trips with gaps >5s: {s['trips_with_gaps_over_5s']}")
        print(f"  GPS-to-road median: {s['gps_to_road_median_m']:.1f}m  |  p95: {s['gps_to_road_p95_m']:.1f}m")
        print(f"  mean two-way fraction: {s['mean_two_way_frac']:.2f}  |  routes fully connected: {s['reference_routes_fully_connected']}/{s['trips']}")
        print()
    print(f"anomalous trips: {summ['anomalous_trips']}")
    print(f"\nplots -> results/eda/plots/   |   maps -> results/maps/eda_*.html")


if __name__ == "__main__":
    main()
