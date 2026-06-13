"""
Usage:
    python train_bert.py \
        --liar_dir   data/liar_dataset \
        --fnn_dir    data/fakenewsnet \
        --model_name bert-base-uncased \
        --epochs     5 \
        --batch_size 32 \
        --lr         2e-5 \
        --output     results/bert
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import load_combined, get_split, print_stats, BERTDataset
from models.bert_model import BERTFakeNewsClassifier, count_params
from utils.trainer import (
    get_device, bert_train_epoch, bert_eval_epoch,
    print_metrics, full_report, EarlyStopping,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--liar_dir",     default="data/liar_dataset")
    p.add_argument("--fnn_dir",      default="data/fakenewsnet")
    p.add_argument("--model_name",   default="bert-base-uncased")
    p.add_argument("--max_len",      type=int,   default=128)
    p.add_argument("--freeze_layers",type=int,   default=0,
                   help="Freeze bottom N transformer layers (0 = fine-tune all)")
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="Fraction of total steps used for linear warmup")
    p.add_argument("--patience",     type=int,   default=3)
    p.add_argument("--output",       default="results/bert")
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

    # ── Tokenizer ──────────────────────────────
    print(f"\n[2] Loading tokenizer: {args.model_name}...")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    # ── Datasets ───────────────────────────────
    print("\n[3] Tokenizing datasets (this may take a moment)...")
    train_ds = BERTDataset(train_df["text"], train_df["binary_label"].values, tokenizer, args.max_len)
    val_ds   = BERTDataset(val_df["text"],   val_df["binary_label"].values,   tokenizer, args.max_len)
    test_ds  = BERTDataset(test_df["text"],  test_df["binary_label"].values,  tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ──────────────────────────────────
    print(f"\n[4] Loading BERT model: {args.model_name}...")
    model = BERTFakeNewsClassifier(
        model_name=args.model_name,
        num_classes=2,
        dropout=args.dropout,
        freeze_layers=args.freeze_layers,
    ).to(device)

    total, trainable = count_params(model)
    print(f"    Total params:     {total:,}")
    print(f"    Trainable params: {trainable:,}")

    # ── Optimizer & Scheduler ──────────────────
    # Separate weight decay: don't apply to bias / LayerNorm weights
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    best_f1 = 0.0
    history = []

    # ── Training loop ──────────────────────────
    print(f"\n[5] Fine-tuning ({args.epochs} epochs, warmup={warmup_steps} steps)...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_m = bert_train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_m = bert_eval_epoch(model, val_loader, criterion, device)
        early_stop.step(val_m["f1_macro"])

        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        print_metrics(train_m, "Train")
        print_metrics(val_m,   "Val  ")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **{f"val_{k}": v for k, v in val_m.items()}})

        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            model.bert.save_pretrained(os.path.join(args.output, "best_bert"))
            torch.save(model.state_dict(), os.path.join(args.output, "best_bert.pt"))
            print(f"    ★ Saved best model (F1={best_f1:.4f})")

        if early_stop.stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break
        print()

    # ── Test evaluation ────────────────────────
    print("\n[6] Test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output, "best_bert.pt"), map_location=device))
    test_loss, test_m = bert_eval_epoch(model, test_loader, criterion, device)

    print_metrics(test_m, "Test")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    cm = full_report(all_labels, all_preds)

    # ── Save results ───────────────────────────
    results = {"model": "BERT", "test_metrics": test_m,
               "history": history, "args": vars(args)}
    with open(os.path.join(args.output, "bert_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
