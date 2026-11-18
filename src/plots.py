"""Model-comparison plots from results/model_comparison.csv, cv_folds.csv, train_histories.json.
Usage: python src/plots.py
"""
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RES  # noqa: E402

P = os.path.join(RES, "plots")
VIO, GRN, GRY = "#5b3fa8", "#1f6b45", "#b8bcb2"


def main():
    comp = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    folds = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
    order = comp.sort_values("cv_macro_f1").model.tolist()

    plt.figure(figsize=(8, 4.5))
    c = comp.set_index("model").loc[order]
    plt.barh(order, c.cv_macro_f1, xerr=c.cv_macro_f1_std, color=VIO, capsize=3)
    plt.xlabel("Macro F1 (mean ± std over 5 GroupKFold folds)"); plt.title("Cross-validation Macro F1")
    plt.xlim(max(0, c.cv_macro_f1.min() - 0.1), 1); plt.tight_layout()
    plt.savefig(os.path.join(P, "cv_macro_f1.png"), dpi=120); plt.close()

    plt.figure(figsize=(9, 4.5))
    folds.boxplot(column="top1", by="model", rot=30, grid=False)
    plt.suptitle(""); plt.title("CV Top-1 accuracy per fold"); plt.ylabel("Top-1"); plt.tight_layout()
    plt.savefig(os.path.join(P, "cv_top1_boxplot.png"), dpi=120); plt.close()

    x = range(len(order))
    plt.figure(figsize=(10, 4.5))
    plt.bar([i - 0.27 for i in x], c.test_top1, 0.27, label="Top-1 (classifier only)", color=GRY)
    plt.bar(list(x), c.test_macro_f1, 0.27, label="Macro F1", color=VIO)
    plt.bar([i + 0.27 for i in x], c.hmm_top1, 0.27, label="Top-1 after HMM", color=GRN)
    plt.xticks(list(x), order, rotation=30); plt.ylim(0.4, 1.0); plt.legend(loc="lower right")
    plt.title("Student-test performance"); plt.tight_layout()
    plt.savefig(os.path.join(P, "student_test_comparison.png"), dpi=120); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.barh(order, c.test_log_loss, color=VIO)
    plt.xlabel("Log loss (student-test, lower is better)"); plt.title("Probability quality")
    plt.xscale("log"); plt.tight_layout(); plt.savefig(os.path.join(P, "log_loss.png"), dpi=120); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.barh(order, c.inference_ms_per_1k_rows, color=VIO)
    plt.xscale("log"); plt.xlabel("Inference time (ms per 1000 candidate rows, log)")
    plt.title("Inference cost"); plt.tight_layout()
    plt.savefig(os.path.join(P, "inference_time.png"), dpi=120); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.scatter(c.inference_ms_per_1k_rows, c.hmm_top1, s=60, color=VIO)
    for m, r in c.iterrows():
        plt.annotate(m, (r.inference_ms_per_1k_rows, r.hmm_top1), fontsize=8, xytext=(4, 4),
                     textcoords="offset points")
    plt.xscale("log"); plt.xlabel("Inference ms / 1k rows"); plt.ylabel("Top-1 after HMM (student-test)")
    plt.title("Accuracy vs inference cost"); plt.tight_layout()
    plt.savefig(os.path.join(P, "accuracy_vs_cost.png"), dpi=120); plt.close()

    h = json.load(open(os.path.join(RES, "train_histories.json")))
    for name, d in h.items():
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(d["train_loss"], color=VIO, label="training loss (log loss)")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
        ax2 = ax1.twinx(); ax2.plot(d["val_score"], color=GRN, label="validation accuracy")
        ax2.set_ylabel("validation accuracy (held-out 10%)")
        fig.legend(loc="center right"); plt.title(f"{name} training history"); fig.tight_layout()
        plt.savefig(os.path.join(P, f"history_{name}.png"), dpi=120); plt.close()
    print("plots written to", P)


if __name__ == "__main__":
    main()
