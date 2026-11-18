"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected classifier (LightGBM).

Objective : mean observation-level Top-1 accuracy over 5-fold GroupKFold on the 80 training trips
            (groups = trips). Macro F1 and log loss per trial are stored as user attributes.
After search: refit with best params on training trips, compare tuned vs untuned on student-test,
and write out-of-fold training probabilities for decoder (HMM) tuning.

Outputs: results/optuna_trials.csv, results/best_params.json, results/tuned_vs_untuned.csv,
         results/plots/optuna_*.png, artifacts/models/final_model.joblib, data/oof_train.pkl
Usage:   python src/tune.py [--trials 40]
"""
import argparse
import os
import time

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ART, RES, ROOT, SEED, core, load_partition, row_metrics, save_json, xy  # noqa: E402
from route_eval import evaluate_partition, summarize  # noqa: E402

UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8)


def make(params):
    return lgb.LGBMClassifier(random_state=SEED, verbose=-1, subsample_freq=1, **params)


def cv_eval(params, X, y, groups, df):
    res = []
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = make(params).fit(X[tr], y[tr])
        res.append(row_metrics(df.iloc[va], m.predict_proba(X[va])[:, 1]))
    return pd.DataFrame(res).mean().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--cv-step", type=int, default=3)
    ap.add_argument("--fit-step", type=int, default=2)
    a = ap.parse_args()

    cv_df = load_partition("train", a.cv_step)
    X, y = xy(cv_df); groups = cv_df.trip.to_numpy()

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        t0 = time.time()
        r = cv_eval(p, X, y, groups, cv_df)
        for k, v in r.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("secs", time.time() - t0)
        print(f"trial {trial.number}: top1={r['top1']:.4f} f1={r['macro_f1']:.4f} ll={r['log_loss']:.4f}", flush=True)
        return r["top1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="pathsnap_lgbm")
    study.enqueue_trial({k: v for k, v in UNTUNED.items() if k != "subsample_freq"} |
                        dict(min_child_samples=20, reg_lambda=1e-3, reg_alpha=1e-3))  # baseline as trial 0
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=a.trials)
    search_secs = time.time() - t0
    tdf = study.trials_dataframe()
    tdf.to_csv(os.path.join(RES, "optuna_trials.csv"), index=False)
    best = study.best_params
    save_json(dict(best_params=best, best_cv_top1=study.best_value, n_trials=len(study.trials),
                   search_secs=search_secs, sampler="TPE", objective="mean 5-fold GroupKFold Top-1"),
              os.path.join(RES, "best_params.json"))

    # plots
    vals = tdf.value.to_numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(vals, "o", ms=4, color="#5b3fa8", alpha=0.6, label="trial")
    plt.plot(np.maximum.accumulate(vals), color="#1f6b45", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV Top-1"); plt.title("Optuna optimisation history"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "optuna_history.png"), dpi=120); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(7, 4))
        plt.barh(list(imp)[::-1], list(imp.values())[::-1], color="#5b3fa8")
        plt.title("Hyperparameter importance (fANOVA)"); plt.tight_layout()
        plt.savefig(os.path.join(RES, "plots", "optuna_importance.png"), dpi=120); plt.close()
    except Exception as e:  # importance needs enough completed trials
        print("importance skipped:", e)

    # tuned vs untuned
    fit_df = load_partition("train", a.fit_step)
    te_df = load_partition("student_test", 1)
    Xf, yf = xy(fit_df); Xt, _ = xy(te_df)
    rows = []
    cache = {}
    for name, p in (("untuned", {k: v for k, v in UNTUNED.items() if k != "subsample_freq"}), ("tuned", best)):
        cvr = cv_eval(p, X, y, groups, cv_df)
        m = make(p); t0 = time.time(); m.fit(Xf, yf); tt = time.time() - t0
        t0 = time.time(); pr = m.predict_proba(Xt)[:, 1]; inf = time.time() - t0
        te = row_metrics(te_df, pr)
        s = summarize(evaluate_partition("student_test", te_df, pr, core.HMMConfig(), cache))
        rows.append(dict(model=f"LightGBM ({name})", cv_top1=cvr["top1"], cv_macro_f1=cvr["macro_f1"],
                         cv_log_loss=cvr["log_loss"], test_top1=te["top1"], test_macro_f1=te["macro_f1"],
                         test_log_loss=te["log_loss"], hmm_top1=s["hmm_top1"], hmm_route_f1=s["hmm_route_f1"],
                         train_secs=tt, inference_ms_per_1k_rows=1e6 * inf / len(Xt)))
        if name == "tuned":
            joblib.dump(m, os.path.join(ART, "models", "final_model.joblib"))
            save_json(dict(model="LightGBM", params=best, features=core.FEATURES,
                           cand_radius=core.CAND_RADIUS, cand_max_k=core.CAND_MAX_K),
                      os.path.join(ART, "model_config.json"))
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)
    print(comp.round(4).T.to_string())

    # out-of-fold probabilities on ALL training points (for decoder tuning without leakage)
    full = load_partition("train", 1)
    Xa, ya = xy(full); ga = full.trip.to_numpy()
    oof = np.zeros(len(full))
    for tr, va in GroupKFold(5).split(Xa, ya, ga):
        oof[va] = make(best).fit(Xa[tr], ya[tr]).predict_proba(Xa[va])[:, 1]
    full.assign(prob=oof).to_pickle(os.path.join(ROOT, "data", "oof_train.pkl"))
    print("OOF top1 on all training points:", row_metrics(full, oof)["top1"])


if __name__ == "__main__":
    main()
