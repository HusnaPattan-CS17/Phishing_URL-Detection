"""
models/svm/train.py
--------------------
Support Vector Machine classifier.
Sweeps kernels (linear, poly, rbf) then tunes C and gamma for rbf.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocessing.data_loader import load_data

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def kernel_sweep(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for kernel in ("linear", "poly", "rbf"):
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            "Kernel":    kernel,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
        })
        print(f"  {kernel:<8}  F1={rows[-1]['F1']:.4f}")
    return pd.DataFrame(rows)


def gamma_c_sweep(X_train, X_test, y_train, y_test,
                  C_values=(0.1, 1, 10),
                  gamma_values=("scale", 0.1, 0.01)) -> pd.DataFrame:
    rows = []
    for C in C_values:
        for gamma in gamma_values:
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rows.append({
                "C": C, "Gamma": gamma,
                "Accuracy":  accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall":    recall_score(y_test, y_pred),
                "F1":        f1_score(y_test, y_pred),
            })
            print(f"  C={C}, gamma={gamma}  F1={rows[-1]['F1']:.4f}")
    return pd.DataFrame(rows)


def train():
    print("Loading data (scaled) …")
    X_train, X_test, y_train, y_test, _ = load_data(scale=True)

    print("\n── Kernel sweep ──")
    kernel_df = kernel_sweep(X_train, X_test, y_train, y_test)
    print(kernel_df.to_string(index=False))

    print("\n── C / Gamma sweep (rbf) ──")
    tuning_df = gamma_c_sweep(X_train, X_test, y_train, y_test)
    print(tuning_df.to_string(index=False))

    best_row = tuning_df.loc[tuning_df["F1"].idxmax()]
    best_C, best_gamma = best_row["C"], best_row["Gamma"]
    print(f"\nBest: C={best_C}, gamma={best_gamma}")

    # ── Final model ───────────────────────────────────────────────────────────
    model = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n── Final Model Performance ──")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Legitimate"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Purples", values_format="d")
    ax.set_title("Confusion Matrix — SVM (rbf)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    kernel_df.to_csv(OUT_DIR / "kernel_sweep.csv", index=False)
    tuning_df.to_csv(OUT_DIR / "tuning_results.csv", index=False)
    print(f"\nArtefacts saved to {OUT_DIR}")
    # ── Kernel comparison graph ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(kernel_df))
    for metric in ("Accuracy", "Precision", "Recall", "F1"):
        axes[0].plot(x, kernel_df[metric], marker="o", linewidth=2, label=metric)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kernel_df["Kernel"])
    axes[0].set_title("SVM — Metrics per Kernel")
    axes[0].set_xlabel("Kernel")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: C vs F1 for rbf (one line per gamma)
    for gval in tuning_df["Gamma"].unique():
        subset = tuning_df[tuning_df["Gamma"] == gval].sort_values("C")
        axes[1].plot(subset["C"].astype(str), subset["F1"],
                     marker="o", linewidth=2, label=f"gamma={gval}")
    axes[1].set_title("SVM (rbf) — C vs F1 per Gamma")
    axes[1].set_xlabel("C")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "hyperparameter_tuning.png", dpi=150)
    plt.close(fig)
    print("  → hyperparameter_tuning.png")
    return model, y_pred, y_test


if __name__ == "__main__":
    train()
