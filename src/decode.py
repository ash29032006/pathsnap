"""HMM/Viterbi decoder tuning and final route-reconstruction evaluation.

1. Tune decoder parameters (beta, emission weight) with Optuna on TRAINING trips, using
   out-of-fold classifier probabilities (data/oof_train.pkl), so the decoder never sees
   probabilities from a model trained on the same trip.
2. Evaluate the final model on the 10 student-test trips: classifier-only vs classifier+HMM.
3. Ablation: HMM on distance-only emissions (no classifier) to show what the classifier adds.
4. Folium maps for every student-test trip: GPS, nearby network, reference route, predicted route.

Outputs: artifacts/hmm_config.json, results/hmm_trials.csv, results/route_eval_final.csv,
         results/hmm_vs_classifier.csv, results/plots/hmm_*.png, results/maps/pred_*.html
Usage:   python src/decode.py [--trials 25] [--tune-trips 30]
"""
import argparse
import os

import folium
import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ART, DATA, RES, ROOT, SEED, core, load_partition, save_json, xy  # noqa: E402
from route_eval import evaluate_partition, summarize  # noqa: E402


def route_latlon(trip, arcs):
    if not arcs:
        return []
    n = trip.nodes; A = trip.arcs
    pts = [[n[A[arcs[0], 0], 1], n[A[arcs[0], 0], 0]]]
    for e in arcs:
        f, t = A[e]
        if [n[f, 1], n[f, 0]] != pts[-1]:
            pts.append(None)  # break in polyline
            pts.append([n[f, 1], n[f, 0]])
        pts.append([n[t, 1], n[t, 0]])
    # split on None
    segs, cur = [], []
    for p in pts:
        if p is None:
            if len(cur) > 1:
                segs.append(cur)
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        segs.append(cur)
    return segs


def pred_map(trip, net, pred_route, title, path):
    lo, la = trip.track[:, 0], trip.track[:, 1]
    m = folium.Map(location=[la.mean(), lo.mean()], zoom_start=14, tiles="OpenStreetMap")
    x0, y0 = trip.xy.min(0) - 150; x1, y1 = trip.xy.max(0) + 150
    near = (net.a[:, 0] > x0) & (net.a[:, 0] < x1) & (net.a[:, 1] > y0) & (net.a[:, 1] < y1)
    fg = folium.FeatureGroup("Road network (nearby)", show=False)
    for e in np.where(near)[0][:6000]:
        f, t = trip.arcs[e]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color="#9aa0a6", weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)
    fr = folium.FeatureGroup("Reference route (green)")
    for s in route_latlon(trip, [int(x) for x in trip.route]):
        folium.PolyLine(s, color="#1f6b45", weight=7, opacity=0.55).add_to(fr)
    fr.add_to(m)
    fp = folium.FeatureGroup("Predicted route (violet)")
    for s in route_latlon(trip, pred_route):
        folium.PolyLine(s, color="#5b3fa8", weight=3.5, opacity=0.95).add_to(fp)
    fp.add_to(m)
    fg2 = folium.FeatureGroup("Raw GPS (blue)")
    for i in range(0, len(lo), max(1, len(lo) // 1500)):
        folium.CircleMarker([la[i], lo[i]], radius=2, color="#2b59c3", fill=True, fill_opacity=0.8,
                            weight=0).add_to(fg2)
    fg2.add_to(m)
    folium.Marker([la[0], lo[0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([la[-1], lo[-1]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    legend = (f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:#fff;padding:8px 12px;'
              f'border:1px solid #b8bcb2;border-radius:6px;font:13px sans-serif">'
              f'<b>{title}</b><br><span style="color:#2b59c3">&#9679;</span> Raw GPS &nbsp;'
              f'<span style="color:#1f6b45">&#9644;</span> Reference &nbsp;'
              f'<span style="color:#5b3fa8">&#9644;</span> Predicted</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[la.min(), lo.min()], [la.max(), lo.max()]])
    m.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--tune-trips", type=int, default=30)
    a = ap.parse_args()

    oof = pd.read_pickle(os.path.join(ROOT, "data", "oof_train.pkl"))
    sizes = oof.groupby("trip").pt.max().sort_values()
    rng = np.random.default_rng(SEED)
    small = sizes[sizes < 6000].index.to_numpy()   # keep tuning time reasonable
    tune_trips = set(rng.choice(small, size=min(a.tune_trips, len(small)), replace=False))
    sub = oof[oof.trip.isin(tune_trips)]
    cache = {}

    def objective(trial):
        cfg = core.HMMConfig(beta=trial.suggest_float("beta", 0.5, 50, log=True),
                             emis_w=trial.suggest_float("emis_w", 0.1, 3.0, log=True))
        r = evaluate_partition("train", sub.drop(columns="prob"), sub.prob.to_numpy(), cfg, cache)
        s = summarize(r)
        trial.set_user_attr("route_f1", s["hmm_route_f1"]); trial.set_user_attr("continuity", s["hmm_continuity"])
        print(f"hmm trial {trial.number}: beta={cfg.beta:.2f} w={cfg.emis_w:.2f} top1={s['hmm_top1']:.4f} "
              f"routeF1={s['hmm_route_f1']:.4f}", flush=True)
        return s["hmm_top1"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(dict(beta=5.0, emis_w=1.0))
    study.optimize(objective, n_trials=a.trials)
    study.trials_dataframe().to_csv(os.path.join(RES, "hmm_trials.csv"), index=False)
    best = core.HMMConfig(**study.best_params)
    save_json(dict(beta=best.beta, emis_w=best.emis_w, break_logp=best.break_logp, max_speed=best.max_speed,
                   tuned_on=sorted(tune_trips), objective="point-weighted HMM Top-1 on training trips (OOF probs)",
                   best_value=study.best_value), os.path.join(ART, "hmm_config.json"))
    print("best decoder config:", best)

    # final evaluation on student-test
    model = joblib.load(os.path.join(ART, "models", "final_model.joblib"))
    te = load_partition("student_test", 1)
    X, _ = xy(te)
    prob = model.predict_proba(X)[:, 1]
    r = evaluate_partition("student_test", te, prob, best, cache)
    r.to_csv(os.path.join(RES, "route_eval_final.csv"), index=False)
    # ablation: distance-only emission (no learned classifier)
    pdist = np.exp(-0.5 * (te.dist.to_numpy() / 8.0) ** 2)
    rd = evaluate_partition("student_test", te, pdist, best, cache)
    s, sd = summarize(r), summarize(rd)
    table = pd.DataFrame([
        dict(method="Nearest arc (distance only, no HMM)", top1=sd["clf_top1"], continuity=sd["clf_continuity"],
             route_f1=sd["clf_route_f1"]),
        dict(method="HMM with distance-only emissions", top1=sd["hmm_top1"], continuity=sd["hmm_continuity"],
             route_f1=sd["hmm_route_f1"]),
        dict(method="Classifier only (LightGBM tuned)", top1=s["clf_top1"], continuity=s["clf_continuity"],
             route_f1=s["clf_route_f1"]),
        dict(method="Classifier + HMM/Viterbi", top1=s["hmm_top1"], continuity=s["hmm_continuity"],
             route_f1=s["hmm_route_f1"]),
    ])
    table.to_csv(os.path.join(RES, "hmm_vs_classifier.csv"), index=False)
    print(table.round(4).to_string())
    print(r[["trip", "n_points", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity", "hmm_route_f1",
             "hmm_breaks"]].round(4).to_string())

    x = np.arange(len(r))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(x - 0.2, r.clf_top1, 0.4, label="classifier only", color="#b8bcb2")
    ax[0].bar(x + 0.2, r.hmm_top1, 0.4, label="classifier + HMM", color="#5b3fa8")
    ax[0].set_title("Per-trip Top-1 accuracy (student-test)")
    ax[1].bar(x - 0.2, r.clf_continuity, 0.4, label="classifier only", color="#b8bcb2")
    ax[1].bar(x + 0.2, r.hmm_continuity, 0.4, label="classifier + HMM", color="#5b3fa8")
    ax[1].set_title("Per-trip route continuity (student-test)")
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(r.trip, rotation=45); a_.set_ylim(0, 1.05); a_.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "hmm_vs_classifier_per_trip.png"), dpi=120); plt.close()

    h = pd.read_csv(os.path.join(RES, "hmm_trials.csv"))
    plt.figure(figsize=(6, 4.5))
    sc = plt.scatter(h.params_beta, h.params_emis_w, c=h.value, cmap="viridis", s=50)
    plt.xscale("log"); plt.yscale("log"); plt.colorbar(sc, label="HMM Top-1 (train OOF)")
    plt.xlabel("beta (m)"); plt.ylabel("emission weight"); plt.title("Decoder parameter search")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "hmm_param_search.png"), dpi=120); plt.close()

    # maps
    te = te.assign(_p=prob)
    for tid, g in te.groupby("trip"):
        trip, net = cache[("student_test", tid)]
        feats = g.drop(columns=["label_arc", "y", "trip", "_p"])
        pred, _ = core.viterbi(trip, net, feats, g._p.to_numpy(), best)
        route = core.assemble_route(net, core.compress(pred))
        row = r[r.trip == tid].iloc[0]
        pred_map(trip, net, route, f"Trip {tid} | Top-1 {row.hmm_top1:.3f} | route F1 {row.hmm_route_f1:.3f}",
                 os.path.join(RES, "maps", f"pred_student_test_{tid}.html"))


if __name__ == "__main__":
    main()
