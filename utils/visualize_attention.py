"""
Usage (after training with --save_attention):
    python -m utils.visualize_attention \
        --weights results/lstm/attention_weights.json \
        --n 10          # show top-10 highest-weight tokens per sample
        --plot          # save PNG heatmaps (requires matplotlib)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_weights(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def top_tokens(record: dict, n: int = 10) -> list[tuple[str, float]]:
    """
    Return the n tokens with highest attention weight, skipping PAD/<unk>.
    """
    skip = {"<pad>", "<unk>", "<s>", "</s>"}
    pairs = [
        (tok, w)
        for tok, w in zip(record["tokens"], record["weights"])
        if tok not in skip
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]


def summary_table(records: list[dict], n: int = 5) -> str:
    """Plain-text table of top attended tokens per record."""
    lines = []
    for i, rec in enumerate(records):
        label = "FAKE" if rec["pred"] == 0 else "REAL"
        tops  = top_tokens(rec, n)
        tok_str = "  |  ".join(f"{t}({w:.3f})" for t, w in tops)
        lines.append(f"[{i:4d}] {label}  {tok_str}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Optional matplotlib plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention(record: dict, out_path: str, max_tokens: int = 30) -> None:
    """
    Save a horizontal bar chart of attention weights for one sample.
    Requires: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[visualize_attention] matplotlib not installed – skipping plots.")
        return

    skip  = {"<pad>", "<unk>"}
    pairs = [(t, w) for t, w in zip(record["tokens"], record["weights"])
             if t not in skip][:max_tokens]
    if not pairs:
        return

    tokens, weights = zip(*pairs)
    y = np.arange(len(tokens))

    fig, ax = plt.subplots(figsize=(8, max(3, len(tokens) * 0.3)))
    bars = ax.barh(y, weights, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attention weight")
    label = "FAKE" if record["pred"] == 0 else "REAL"
    ax.set_title(f"Predicted: {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser("Visualize BiLSTM attention weights")
    p.add_argument("--weights", required=True,
                   help="Path to attention_weights.json")
    p.add_argument("--n",       type=int, default=10,
                   help="Top-N tokens to show per sample")
    p.add_argument("--max",     type=int, default=20,
                   help="Max samples to display in the table")
    p.add_argument("--plot",    action="store_true",
                   help="Save per-sample PNG heatmaps")
    p.add_argument("--plot_dir", default="results/attention_plots",
                   help="Directory for PNG outputs")
    return p.parse_args()


def main():
    args    = _parse()
    records = load_weights(args.weights)

    # Print table
    print(summary_table(records[: args.max], n=args.n))

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        for i, rec in enumerate(records[: args.max]):
            out = os.path.join(args.plot_dir, f"sample_{i:04d}.png")
            plot_attention(rec, out)
        print(f"\nPlots saved to {args.plot_dir}/")


if __name__ == "__main__":
    main()