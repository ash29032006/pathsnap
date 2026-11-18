"""Train and compare candidate classifiers from several algorithm families.

Each row = (GPS point, candidate arc); target y = 1 if the candidate is the reference arc.
- 5-fold GroupKFold on the 80 training trips (groups = trip, so no trip is split across folds)
- refit on all training trips, evaluate on the 10 student-test trips
- route-level evaluation (classifier-only and with default HMM) on student-test

Outputs: results/cv_folds.csv, results/model_comparison.csv, artifacts/models/*.joblib,
         results/plots/*, results/train_histories.json
Usage:   python src/train_models.py [--cv-step 3] [--fit-step 2]
"""
import argparse
import time

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ART, RES, SEED, core, load_partition, row_metrics, save_json, xy  # noqa: E402
import os  # noqa: E402
from route_eval import evaluate_partition, summarize  # noqa: E402


def models():
    return {
        "LogisticRegression": ("Linear", "C=1.0, lbfgs, standardised",
                               lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
        "GaussianNB": ("Probabilistic", "var_smoothing=1e-9",
                       lambda: GaussianNB()),
        "KNN": ("Instance-based", "k=25, distance-weighted, standardised",
                lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
        "DecisionTree": ("Tree", "max_depth=10, min_samples_leaf=20",
                         lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=SEED)),
        "RandomForest": ("Ensemble (bagging)", "200 trees, max_depth=16, min_samples_leaf=5",
                         lambda: RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                                        n_jobs=-1, random_state=SEED)),
        "ExtraTrees": ("Ensemble (randomised)", "300 trees, max_depth=18, min_samples_leaf=5",
                       lambda: ExtraTreesClassifier(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                                    n_jobs=-1, random_state=SEED)),
        "LightGBM": ("Boosting", "400 rounds, lr=0.05, 63 leaves",
                     lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                random_state=SEED, verbose=-1)),
        "MLP": ("Neural network", "(64,32) ReLU, adam, early stopping",
                lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                      batch_size=512, max_iter=60, early_stopping=True,
                                                                      validation_fraction=0.1, n_iter_no_change=6,
                                                                      random_state=SEED))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-step", type=int, default=3, help="keep every k-th GPS point for CV")
    ap.add_argument("--fit-step", type=int, default=2, help="keep every k-th GPS point for the final fit")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    cv_df = load_partition("train", a.cv_step)
    fit_df = load_partition("train", a.fit_step)
    te_df = load_partition("student_test", 1)
    print(f"cv rows={len(cv_df)} fit rows={len(fit_df)} test rows={len(te_df)} pos-rate={cv_df.y.mean():.3f}")
    Xcv, ycv = xy(cv_df); groups = cv_df.trip.to_numpy()
    Xfit, yfit = xy(fit_df); Xte, yte = xy(te_df)
    gkf = GroupKFold(n_splits=5)

    fold_rows, comp_rows, hist = [], [], {}
    cache = {}
    specs = models()
    if a.only:
        specs = {k: v for k, v in specs.items() if k in a.only}
    for name, (family, cfg, make) in specs.items():
        print(f"== {name}", flush=True)
        for k, (tr, va) in enumerate(gkf.split(Xcv, ycv, groups)):
            m = make(); t0 = time.time(); m.fit(Xcv[tr], ycv[tr]); ft = time.time() - t0
            p = m.predict_proba(Xcv[va])[:, 1]
            r = row_metrics(cv_df.iloc[va], p)
            fold_rows.append(dict(model=name, fold=k, fit_secs=ft, **r))
            print(f"   fold {k}: top1={r['top1']:.4f} f1={r['macro_f1']:.4f} ll={r['log_loss']:.4f}", flush=True)
        f = pd.DataFrame([x for x in fold_rows if x["model"] == name])

        m = make(); t0 = time.time(); m.fit(Xfit, yfit); train_secs = time.time() - t0
        t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf = time.time() - t0
        joblib.dump(m, os.path.join(ART, "models", f"{name}.joblib"))
        te = row_metrics(te_df, p)
        rr = evaluate_partition("student_test", te_df, p, core.HMMConfig(), cache)
        s = summarize(rr)
        rr.to_csv(os.path.join(RES, f"route_eval_student_test_{name}.csv"), index=False)
        cm = confusion_matrix(yte, (p >= 0.5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref arc"]).plot(cmap="Blues", values_format="d")
        plt.title(f"{name} - student-test confusion matrix"); plt.tight_layout()
        plt.savefig(os.path.join(RES, "plots", f"cm_{name}.png"), dpi=120); plt.close()

        last = m[-1] if hasattr(m, "steps") else m
        if isinstance(last, MLPClassifier):
            hist[name] = dict(train_loss=list(last.loss_curve_), val_score=list(last.validation_scores_))
        comp_rows.append(dict(
            model=name, family=family, config=cfg,
            cv_macro_f1=f.macro_f1.mean(), cv_macro_f1_std=f.macro_f1.std(), cv_top1=f.top1.mean(),
            cv_log_loss=f.log_loss.mean(),
            test_top1=te["top1"], test_macro_f1=te["macro_f1"], test_precision=te["precision"],
            test_recall=te["recall"], test_log_loss=te["log_loss"],
            clf_route_continuity=s["clf_continuity"], hmm_top1=s["hmm_top1"],
            hmm_route_continuity=s["hmm_continuity"], hmm_route_f1=s["hmm_route_f1"],
            train_secs=train_secs, inference_ms_per_1k_rows=1000 * inf / len(Xte) * 1000,
        ))
        print(pd.Series(comp_rows[-1]).to_string(), flush=True)

    pd.DataFrame(fold_rows).to_csv(os.path.join(RES, "cv_folds.csv"), index=False)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    save_json(hist, os.path.join(RES, "train_histories.json"))
    print(comp[["model", "cv_macro_f1", "test_top1", "test_macro_f1", "test_log_loss", "hmm_top1",
                "hmm_route_continuity", "inference_ms_per_1k_rows"]].round(4).to_string())


if __name__ == "__main__":
    main()
