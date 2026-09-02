"""
models/logistic_regression/train.py
-------------------------------------
Trains a Logistic Regression classifier for phishing URL detection.

Hyperparameter sweep over regularisation strength C, selects the best
model by F1-score, and persists evaluation artefacts.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocessing.data_loader import load_data

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def sweep(X_train, X_test, y_train, y_test,
          C_values=(0.01, 0.1, 1, 10)) -> pd.DataFrame:
    rows = []
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            "C": C,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
        })
        print(f"  C={C:<6}  F1={rows[-1]['F1']:.4f}")
    return pd.DataFrame(rows)


def train():
    print("Loading data …")
    X_train, X_test, y_train, y_test, _ = load_data(scale=True)

    print("\n── Hyperparameter sweep (C) ──")
    results = sweep(X_train, X_test, y_train, y_test)
    print(results.to_string(index=False))

    best_C = results.loc[results["F1"].idxmax(), "C"]
    print(f"\nBest C = {best_C}")

    # ── Final model ───────────────────────────────────────────────────────────
    model = LogisticRegression(C=best_C, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n── Final Model Performance ──")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Legitimate"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Logistic Regression")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── Sweep plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ("Accuracy", "Precision", "Recall", "F1"):
        ax.plot(results["C"].astype(str), results[metric], marker="o", label=metric)
    ax.set_title("Logistic Regression — C sweep")
    ax.set_xlabel("C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "c_sweep.png", dpi=150)
    plt.close(fig)

    results.to_csv(OUT_DIR / "sweep_results.csv", index=False)
    print(f"\nArtefacts saved to {OUT_DIR}")
    # ── Hyperparameter tuning graph ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all 4 metrics vs C
    for metric in ("Accuracy", "Precision", "Recall", "F1"):
        axes[0].plot(results["C"].astype(str), results[metric],
                     marker="o", linewidth=2, label=metric)
    axes[0].set_title("Logistic Regression — Metrics vs C")
    axes[0].set_xlabel("Regularisation Strength (C)")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: highlight best C with a bar chart of its metrics
    best_row = results.loc[results["F1"].idxmax()]
    metrics_vals = [best_row["Accuracy"], best_row["Precision"],
                    best_row["Recall"], best_row["F1"]]
    bars = axes[1].bar(["Accuracy", "Precision", "Recall", "F1"],
                       metrics_vals, color=["steelblue","coral","mediumseagreen","gold"])
    axes[1].set_ylim(0.85, 1.02)
    axes[1].set_title(f"Best Model Metrics (C={best_row['C']})")
    axes[1].set_ylabel("Score")
    for bar, val in zip(bars, metrics_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "hyperparameter_tuning.png", dpi=150)
    plt.close(fig)
    print("  → hyperparameter_tuning.png")
    return model, y_pred, y_test


if __name__ == "__main__":
    train()
