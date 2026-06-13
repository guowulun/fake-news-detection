"""
Usage:
    python compare_models.py \
        --lstm_results results/lstm/lstm_results.json \
        --bert_results results/bert/bert_results.json \
        --output       results/comparison
"""

import os
import json
import argparse
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def print_comparison(lstm_r, bert_r):
    metrics = ["accuracy", "f1_macro", "precision", "recall"]
    lstm_m = lstm_r["test_metrics"]
    bert_m = bert_r["test_metrics"]

    print("\n" + "═" * 58)
    print(f"{'Metric':<16}  {'BiLSTM':>10}  {'BERT':>10}  {'Δ (BERT-LSTM)':>14}")
    print("─" * 58)
    for m in metrics:
        lv = lstm_m[m]
        bv = bert_m[m]
        delta = bv - lv
        arrow = "▲" if delta > 0 else "▼"
        print(f"{m:<16}  {lv:>10.4f}  {bv:>10.4f}  {arrow}{abs(delta):>13.4f}")
    print("═" * 58)

    winner = "BERT" if bert_m["f1_macro"] > lstm_m["f1_macro"] else "BiLSTM"
    print(f"\n  Best model by F1-macro: {winner}")
    improvement = abs(bert_m["f1_macro"] - lstm_m["f1_macro"]) / max(lstm_m["f1_macro"], 1e-9) * 100
    print(f"  Relative improvement  : {improvement:.1f}%\n")


def save_summary(lstm_r, bert_r, output_dir):
    summary = {
        "lstm": {
            "model": lstm_r["model"],
            "test": lstm_r["test_metrics"],
            "best_val_f1": max(e["val_f1_macro"] for e in lstm_r["history"]),
            "epochs_trained": len(lstm_r["history"]),
        },
        "bert": {
            "model": bert_r["model"],
            "test": bert_r["test_metrics"],
            "best_val_f1": max(e["val_f1_macro"] for e in bert_r["history"]),
            "epochs_trained": len(bert_r["history"]),
        },
    }
    path = os.path.join(output_dir, "comparison_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {path}")


def plot_training_curves(lstm_r, bert_r, output_dir):
    """Generate ASCII training curves (no matplotlib dependency required)."""
    print("\n── Validation F1 per epoch (ASCII) ──\n")

    for label, r in [("BiLSTM", lstm_r), ("BERT", bert_r)]:
        f1s = [e["val_f1_macro"] for e in r["history"]]
        print(f"  {label}:")
        for i, v in enumerate(f1s, 1):
            bar = "█" * int(v * 40)
            print(f"    Ep{i:02d} {bar} {v:.4f}")
        print()

    # Try matplotlib if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for label, r, color in [
            ("BiLSTM", lstm_r, "#E8593C"),
            ("BERT",   bert_r, "#534AB7"),
        ]:
            epochs = [e["epoch"] for e in r["history"]]

            axes[0].plot(epochs, [e["train_loss"] for e in r["history"]],
                         label=f"{label} train", color=color, linewidth=2)
            axes[0].plot(epochs, [e["val_loss"] for e in r["history"]],
                         label=f"{label} val", color=color, linewidth=2, linestyle="--")

            axes[1].plot(epochs, [e["val_f1_macro"] for e in r["history"]],
                         label=label, color=color, linewidth=2, marker="o")

        for ax, title, ylabel in [
            (axes[0], "Training & Validation Loss", "Loss"),
            (axes[1], "Validation F1-macro",        "F1-macro"),
        ]:
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {path}")

    except ImportError:
        print("  (matplotlib not installed, skipping plot)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lstm_results", default="results/lstm/lstm_results.json")
    p.add_argument("--bert_results", default="results/bert/bert_results.json")
    p.add_argument("--output",       default="results/comparison")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading results...")
    lstm_r = load_results(args.lstm_results)
    bert_r = load_results(args.bert_results)

    print_comparison(lstm_r, bert_r)
    save_summary(lstm_r, bert_r, args.output)
    plot_training_curves(lstm_r, bert_r, args.output)


if __name__ == "__main__":
    main()
