"""Log every PathSnap experiment into MLflow, for browsing in the MLflow UI during the demo.

Nothing is retrained. The script reads the result files that train_models.py, tune.py and
decode.py already wrote, so every number in MLflow is the same number as in results/.

Experiments created:
  1 - Model comparison        one run per classifier (8): params, CV + student-test metrics,
                              per-fold metrics (step = fold), MLP loss curve (step = epoch),
                              confusion matrix, per-trip route evaluation, model file
  2 - Bayesian tuning         parent run for the Optuna TPE search with one child run per trial,
                              plus "LightGBM untuned" vs "LightGBM tuned" runs; the tuned model is
                              registered as "PathSnap-LightGBM"
  3 - HMM decoding            parent run for the HMM parameter search with one child run per trial,
                              plus one run per decoding method (nearest arc, HMM distance-only,
                              classifier only, classifier + HMM) with Folium prediction maps

Usage:   python src/log_mlflow.py            (rebuilds mlflow.db and mlruns/ from scratch)
         mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
"""
import io
import json
import logging
import os
import shutil
import subprocess
import warnings

import joblib
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402

from common import ART, RES, ROOT  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

DB = os.path.join(ROOT, "mlflow.db")
RUNS = os.path.join(ROOT, "mlruns")
PLOTS = os.path.join(RES, "plots")
MAPS = os.path.join(RES, "maps")
SELECTED = "LightGBM"
# Model files bigger than this are referenced by path instead of copied into MLflow
# (None = log every model; KNN / RandomForest / ExtraTrees add ~520 MB to mlruns/).
MAX_MODEL_MB = None


def committed(rel: str):
    """Contents of a results file as committed in git, or None."""
    try:
        return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True,
                              check=True, text=True).stdout
    except Exception:
        return None


def load_cv_folds(models) -> pd.DataFrame:
    """Per-fold CV metrics. A partial re-run (train_models.py --only ...) rewrites cv_folds.csv
    with only the re-run models, so fill any missing model from the committed copy."""
    df = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
    missing = set(models) - set(df.model)
    if missing:
        txt = committed("results/cv_folds.csv")
        if txt:
            old = pd.read_csv(io.StringIO(txt))
            df = pd.concat([df, old[old.model.isin(missing)]], ignore_index=True)
            print(f"  cv_folds.csv: filled {sorted(missing)} from the committed copy")
    return df


def load_histories() -> dict:
    with open(os.path.join(RES, "train_histories.json")) as f:
        hist = json.load(f)
    if not hist:
        txt = committed("results/train_histories.json")
        if txt:
            hist = json.loads(txt)
            print("  train_histories.json: empty, using the committed copy")
    return hist


def flat_params(model) -> dict:
    """Constructor parameters of an estimator / pipeline, without nested estimator objects."""
    out = {}
    for k, v in model.get_params(deep=True).items():
        if hasattr(v, "get_params") or isinstance(v, (list, tuple)) and v and hasattr(v[0], "__len__"):
            continue
        out[k] = str(v)[:500]
    return out


def log_if(path, artifact_path=None):
    if os.path.exists(path):
        mlflow.log_artifact(path, artifact_path)


def log_model(model_file, name, registered=None):
    size_mb = os.path.getsize(model_file) / 1e6
    mlflow.set_tag("model_file", os.path.relpath(model_file, ROOT))
    mlflow.set_tag("model_size_mb", f"{size_mb:.1f}")
    if MAX_MODEL_MB is not None and size_mb > MAX_MODEL_MB:
        mlflow.set_tag("model_logged", f"no (>{MAX_MODEL_MB} MB, kept at model_file)")
        return
    # cloudpickle, like the joblib files themselves (these are our own trained models)
    mlflow.sklearn.log_model(joblib.load(model_file), name=name, registered_model_name=registered,
                             serialization_format="cloudpickle")
    mlflow.set_tag("model_logged", "yes")


def experiment(name, description):
    exp_id = mlflow.create_experiment(name, artifact_location=f"file://{RUNS}",
                                      tags={"mlflow.note.content": description})
    mlflow.set_experiment(experiment_id=exp_id)


# --------------------------------------------------------------------------------------------
def log_model_comparison():
    experiment("1 - Model comparison",
               "8 classifiers scoring (GPS point, candidate arc) pairs. 5-fold GroupKFold CV on "
               "training trips (grouped by trip) + evaluation on the 10 student-test trips. "
               "Route metrics use the default HMM config. Source: src/train_models.py.")
    comp = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    folds = load_cv_folds(comp.model)
    hist = load_histories()
    for _, r in comp.iterrows():
        name = r.model
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"family": r.family, "stage": "model_comparison",
                             "selected_model": str(name == SELECTED).lower()})
            mlflow.log_param("config", r.config)
            mf = os.path.join(ART, "models", f"{name}.joblib")
            if os.path.exists(mf):
                mlflow.log_params(flat_params(joblib.load(mf)))
            metrics = {k: float(r[k]) for k in comp.columns if k not in ("model", "family", "config")}
            mlflow.log_metrics(metrics)
            for _, f in folds[folds.model == name].iterrows():
                for k in ("top1", "macro_f1", "precision", "recall", "log_loss", "fit_secs"):
                    mlflow.log_metric(f"fold_{k}", float(f[k]), step=int(f.fold))
            if name in hist:
                for k, series in hist[name].items():
                    for i, v in enumerate(series):
                        mlflow.log_metric(f"epoch_{k}", float(v), step=i + 1)
                log_if(os.path.join(PLOTS, f"history_{name}.png"), "plots")
            log_if(os.path.join(PLOTS, f"cm_{name}.png"), "plots")
            log_if(os.path.join(RES, f"route_eval_student_test_{name}.csv"), "route_eval")
            if os.path.exists(mf):
                log_model(mf, "model")
        print(f"  {name}: test macro F1 {r.test_macro_f1:.4f}")

    # one summary run holding the cross-model comparison plots
    with mlflow.start_run(run_name="Comparison plots"):
        mlflow.set_tag("stage", "summary")
        for p in ("cv_macro_f1", "cv_top1_boxplot", "student_test_comparison", "log_loss",
                  "inference_time", "accuracy_vs_cost"):
            log_if(os.path.join(PLOTS, f"{p}.png"), "plots")
        log_if(os.path.join(RES, "model_comparison.csv"))


def log_tuning():
    with open(os.path.join(RES, "best_params.json")) as f:
        best = json.load(f)
    experiment("2 - Bayesian tuning (LightGBM)",
               f"Optuna {best['sampler']} search, {best['n_trials']} trials. Objective: "
               f"{best['objective']}. Source: src/tune.py.")
    trials = pd.read_csv(os.path.join(RES, "optuna_trials.csv"))
    pcols = [c for c in trials.columns if c.startswith("params_")]
    with mlflow.start_run(run_name="Optuna TPE search"):
        mlflow.set_tags({"stage": "tuning", "sampler": best["sampler"]})
        mlflow.log_params({"n_trials": best["n_trials"], "objective": best["objective"]})
        mlflow.log_params({f"best_{k}": v for k, v in best["best_params"].items()})
        mlflow.log_metrics({"best_cv_top1": best["best_cv_top1"], "search_secs": best["search_secs"]})
        running = 0.0
        for _, t in trials.iterrows():
            running = max(running, t.value)
            mlflow.log_metric("trial_cv_top1", float(t.value), step=int(t.number))
            mlflow.log_metric("best_so_far_cv_top1", running, step=int(t.number))
            with mlflow.start_run(run_name=f"trial {int(t.number):02d}", nested=True):
                mlflow.set_tag("stage", "tuning_trial")
                mlflow.log_params({c.replace("params_", ""): t[c] for c in pcols})
                mlflow.log_metrics({"cv_top1": float(t.value),
                                    "cv_macro_f1": float(t.user_attrs_macro_f1),
                                    "cv_log_loss": float(t.user_attrs_log_loss),
                                    "cv_precision": float(t.user_attrs_precision),
                                    "cv_recall": float(t.user_attrs_recall),
                                    "trial_secs": float(t.user_attrs_secs)})
        for p in ("optuna_history", "optuna_importance"):
            log_if(os.path.join(PLOTS, f"{p}.png"), "plots")
        log_if(os.path.join(RES, "optuna_trials.csv"))
        log_if(os.path.join(RES, "best_params.json"))
    print(f"  Optuna: {len(trials)} trials")

    tvu = pd.read_csv(os.path.join(RES, "tuned_vs_untuned.csv"))
    for _, r in tvu.iterrows():
        tuned = "(tuned)" in r.model
        with mlflow.start_run(run_name=r.model):
            mlflow.set_tags({"stage": "tuned_vs_untuned", "selected_model": str(tuned).lower()})
            mlflow.log_metrics({k: float(r[k]) for k in tvu.columns if k != "model"})
            mf = os.path.join(ART, "models", "final_model.joblib" if tuned else "LightGBM.joblib")
            mlflow.log_params(flat_params(joblib.load(mf)))
            if tuned:
                log_if(os.path.join(ART, "model_config.json"))
                log_model(mf, "model", registered="PathSnap-LightGBM")
            else:
                log_model(mf, "model")
        print(f"  {r.model}: test macro F1 {r.test_macro_f1:.4f}")


def log_hmm():
    with open(os.path.join(ART, "hmm_config.json")) as f:
        cfg = json.load(f)
    experiment("3 - HMM decoding",
               f"HMM/Viterbi route reconstruction on top of the tuned LightGBM. Decoder params "
               f"tuned with Optuna; objective: {cfg['objective']}. Evaluated on the 10 "
               f"student-test trips. Source: src/decode.py.")
    trials = pd.read_csv(os.path.join(RES, "hmm_trials.csv"))
    with mlflow.start_run(run_name="HMM parameter search"):
        mlflow.set_tag("stage", "hmm_tuning")
        mlflow.log_params({"n_trials": len(trials), "objective": cfg["objective"],
                           "n_tuning_trips": len(cfg["tuned_on"])})
        mlflow.log_params({f"best_{k}": cfg[k] for k in ("beta", "emis_w", "break_logp", "max_speed")})
        mlflow.log_metric("best_value", cfg["best_value"])
        for _, t in trials.iterrows():
            mlflow.log_metric("trial_hmm_top1", float(t.value), step=int(t.number))
            with mlflow.start_run(run_name=f"trial {int(t.number):02d}", nested=True):
                mlflow.set_tag("stage", "hmm_trial")
                mlflow.log_params({"beta": t.params_beta, "emis_w": t.params_emis_w})
                mlflow.log_metrics({"hmm_top1": float(t.value),
                                    "continuity": float(t.user_attrs_continuity),
                                    "route_f1": float(t.user_attrs_route_f1)})
        log_if(os.path.join(PLOTS, "hmm_param_search.png"), "plots")
        log_if(os.path.join(RES, "hmm_trials.csv"))
        log_if(os.path.join(ART, "hmm_config.json"))
    print(f"  HMM search: {len(trials)} trials")

    hvc = pd.read_csv(os.path.join(RES, "hmm_vs_classifier.csv"))
    for _, r in hvc.iterrows():
        final = r.method.startswith("Classifier + HMM")
        with mlflow.start_run(run_name=r.method):
            mlflow.set_tags({"stage": "decoding_comparison", "selected_model": str(final).lower()})
            mlflow.log_metrics({"student_test_top1": float(r.top1),
                                "student_test_continuity": float(r.continuity),
                                "student_test_route_f1": float(r.route_f1)})
            if final:
                mlflow.log_params({k: cfg[k] for k in ("beta", "emis_w", "break_logp", "max_speed")})
                log_if(os.path.join(ART, "hmm_config.json"))
                log_if(os.path.join(RES, "route_eval_final.csv"))
                log_if(os.path.join(RES, "hmm_vs_classifier.csv"))
                log_if(os.path.join(PLOTS, "hmm_vs_classifier_per_trip.png"), "plots")
                for m in sorted(os.listdir(MAPS)):
                    if m.startswith("pred_"):
                        mlflow.log_artifact(os.path.join(MAPS, m), "maps")
        print(f"  {r.method}: top-1 {r.top1:.4f}")


def main():
    for p in (DB, RUNS):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    print("Model comparison")
    log_model_comparison()
    print("Bayesian tuning")
    log_tuning()
    print("HMM decoding")
    log_hmm()
    ev = os.path.join(RES, "evaluator_test")
    if os.path.exists(os.path.join(ev, "per_trip.csv")):  # written by src/eval_evaluator.py
        from eval_evaluator import log_to_mlflow
        with open(os.path.join(ev, "summary.json")) as f:
            summary = json.load(f)
        log_to_mlflow(pd.read_csv(os.path.join(ev, "per_trip.csv"), dtype={"trip": str}), summary)
        print("Evaluator test: re-logged from results/evaluator_test/")
    print(f"\nDone. Start the UI with:\n  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002")


if __name__ == "__main__":
    main()
