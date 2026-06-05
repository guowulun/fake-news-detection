"""
Train LSTM baseline on LIAR + FakeNewsNet.

Usage:
    python train_lstm.py \
        --liar_dir  data/liar_dataset \
        --fnn_dir   data/fakenewsnet \
        --glove     data/glove.6B.300d.txt \
        --epochs    20 \
        --batch_size 64 \
        --lr        1e-3 \
        --output    results/lstm
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import load_combined, get_split, print_stats, LSTMDataset
from data.vocab import build_vocab, load_glove
from models.lstm_model import BiLSTMClassifier, count_params
from utils.trainer import (
    get_device, lstm_train_epoch, lstm_eval_epoch,
    print_metrics, full_report, EarlyStopping,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--liar_dir",   default="data/liar_dataset")
    p.add_argument("--fnn_dir",    default="data/fakenewsnet")
    p.add_argument("--glove",      default="data/glove.6B.300d.txt")
    p.add_argument("--max_len",    type=int, default=128)
    p.add_argument("--min_freq",   type=int, default=2)
    p.add_argument("--embed_dim",  type=int, default=300)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--freeze_emb", action="store_true", default=True)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int, default=5)
    p.add_argument("--output",     default="results/lstm")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = get_device()

    # ── Load data ──────────────────────────────
    print("\n[1] Loading datasets...")
    df = load_combined(args.liar_dir, args.fnn_dir)
    print_stats(df)

    train_df = get_split(df, "train")
    val_df   = get_split(df, "valid")
    test_df  = get_split(df, "test")

    # ── Build vocab ────────────────────────────
    print("\n[2] Building vocabulary...")
    vocab = build_vocab(train_df["text"], min_freq=args.min_freq)

    # ── Load GloVe ─────────────────────────────
    print("\n[3] Loading GloVe embeddings...")
    embed_matrix = load_glove(args.glove, vocab, embed_dim=args.embed_dim)

    # ── Datasets & Loaders ─────────────────────
    print("\n[4] Creating datasets...")
    train_ds = LSTMDataset(train_df["text"], train_df["binary_label"].values, vocab, args.max_len)
    val_ds   = LSTMDataset(val_df["text"],   val_df["binary_label"].values,   vocab, args.max_len)
    test_ds  = LSTMDataset(test_df["text"],  test_df["binary_label"].values,  vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ──────────────────────────────────
    print("\n[5] Building LSTM model...")
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        pretrained_embeddings=embed_matrix,
        freeze_embeddings=args.freeze_emb,
    ).to(device)
    print(f"    Trainable params: {count_params(model):,}")

    # ── Training setup ─────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    best_f1 = 0.0
    history = []

    # ── Training loop ──────────────────────────
    print("\n[6] Training...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_m = lstm_train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_m   = lstm_eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_m["f1_macro"])
        early_stop.step(val_m["f1_macro"])

        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        print_metrics(train_m, "Train")
        print_metrics(val_m,   "Val  ")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **{f"val_{k}": v for k, v in val_m.items()}})

        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            torch.save(model.state_dict(), os.path.join(args.output, "best_lstm.pt"))
            print(f"    ★ Saved best model (F1={best_f1:.4f})")

        if early_stop.stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break
        print()

    # ── Test evaluation ────────────────────────
    print("\n[7] Test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output, "best_lstm.pt"), map_location=device))
    test_loss, test_m = lstm_eval_epoch(model, test_loader, criterion, device)

    print_metrics(test_m, "Test")

    # Collect predictions for full report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, labels in test_loader:
            logits = model(input_ids.to(device))
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = full_report(all_labels, all_preds)

    # ── Save results ───────────────────────────
    results = {"model": "BiLSTM", "test_metrics": test_m,
               "history": history, "args": vars(args)}
    with open(os.path.join(args.output, "lstm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
