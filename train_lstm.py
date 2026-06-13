"""
Usage
-----
python train_lstm.py \
    --liar_dir  data/liar_dataset \
    --fnn_dir   data/fakenewsnet \
    --glove     data/glove.6B.300d.txt \
    --epochs    20 \
    --batch_size 64 \
    --lr        1e-3 \
    --output    results/lstm \
    --save_attention          # optional: dump attention weights on test set
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project imports (fixed) ───────────────────────────────────────────────────
from data.dataset import load_combined, get_split, print_stats, LSTMDataset
from data.vocab   import build_vocab, load_glove
from models.lstm_model import build_model
from utils.trainer import (
    get_device,
    lstm_train_epoch,
    lstm_eval_epoch,
    print_metrics,
    full_report,
    EarlyStopping,
)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("BiLSTM + Attention – Fake News Detection")
    p.add_argument("--liar_dir",       required=True)
    p.add_argument("--fnn_dir",        required=True)
    p.add_argument("--glove",          required=True)
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--num_layers",     type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--max_len",        type=int,   default=128)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--output",         default="results/lstm")
    p.add_argument("--no_attention",   action="store_true",
                   help="Disable attention (fall back to mean-pooling).")
    p.add_argument("--save_attention", action="store_true",
                   help="Save per-sample attention weights after evaluation.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def dump_attention_weights(model, loader, vocab, device, out_path: str):
    """
    Run the test set through the model with return_attention=True and
    write a JSON file:  [{tokens: [...], weights: [...], pred: 0|1}, ...]
    """
    model.eval()
    records = []
    # FIX: vocab là dict {word: idx}, đảo ngược để idx -> word
    idx2word = {v: k for k, v in vocab.items()}

    with torch.no_grad():
        for input_ids, labels in loader:
            ids = input_ids.to(device)
            logits, attn = model(ids, return_attention=True)   # (B,C), (B,T)
            preds = logits.argmax(dim=-1).cpu().tolist()

            for i in range(ids.size(0)):
                tokens  = [idx2word.get(t.item(), "<unk>") for t in ids[i]]
                weights = attn[i].cpu().tolist() if attn is not None else []
                records.append({"tokens": tokens, "weights": weights,
                                "pred": preds[i]})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[attention] saved {len(records)} samples → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    os.makedirs(args.output, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    # FIX: load_liar/load_fakenewsnet trả về DataFrame, không phải tuples.
    # Dùng load_combined() + get_split() cho đúng API.
    print("\n[1] Loading datasets...")
    df = load_combined(args.liar_dir, args.fnn_dir)
    print_stats(df)

    train_df = get_split(df, "train")
    val_df   = get_split(df, "valid")
    test_df  = get_split(df, "test")

    train_texts  = train_df["text"].tolist()
    train_labels = train_df["binary_label"].tolist()
    val_texts    = val_df["text"].tolist()
    val_labels   = val_df["binary_label"].tolist()
    test_texts   = test_df["text"].tolist()
    test_labels  = test_df["binary_label"].tolist()

    # ── Vocabulary + GloVe ───────────────────────────────────────────────────
    # FIX: build_vocab() và load_glove() là functions, không phải class Vocabulary
    print("\n[2] Building vocabulary...")
    vocab = build_vocab(train_texts)                      # dict {word: idx}

    print(f"\n[3] Loading GloVe: {args.glove}...")
    glove_matrix = load_glove(args.glove, vocab)          # numpy (vocab_size, embed_dim)
    embed_dim    = glove_matrix.shape[1]

    # ── Datasets / Loaders ───────────────────────────────────────────────────
    # FIX: LSTMDataset thay vì FakeNewsDataset (tên không tồn tại)
    print("\n[4] Creating datasets...")
    train_ds = LSTMDataset(train_texts, train_labels, vocab, args.max_len)
    val_ds   = LSTMDataset(val_texts,   val_labels,   vocab, args.max_len)
    test_ds  = LSTMDataset(test_texts,  test_labels,  vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ────────────────────────────────────────────────────────────────
    use_attention = not args.no_attention
    print(f"\n[5] Building model: BiLSTM {'+ Attention' if use_attention else '(mean-pool)'}...")
    model = build_model(
        vocab_size    = len(vocab),
        embed_dim     = embed_dim,
        hidden_dim    = args.hidden_dim,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
        use_attention = use_attention,
    ).to(device)

    # FIX: load_pretrained_embeddings nhận torch.Tensor, không phải numpy array
    model.load_pretrained_embeddings(torch.tensor(glove_matrix))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {trainable:,}")

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion  = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    best_f1 = 0.0
    history = []

    # ── Training loop ─────────────────────────────────────────────────────────
    # FIX: dùng lstm_train_epoch/lstm_eval_epoch từ utils.trainer
    # thay vì class Trainer (không tồn tại)
    print(f"\n[6] Training ({args.epochs} epochs)...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_m = lstm_train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_m   = lstm_eval_epoch(model, val_loader, criterion, device)
        early_stop.step(val_m["f1_macro"])

        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        print_metrics(train_m, "Train")
        print_metrics(val_m,   "Val  ")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            **{f"val_{k}": v for k, v in val_m.items()},
        })

        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            torch.save(model.state_dict(), os.path.join(args.output, "best_lstm.pt"))
            print(f"    ★ Saved best model (F1={best_f1:.4f})")

        if early_stop.stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break
        print()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n[7] Test evaluation...")
    model.load_state_dict(
        torch.load(os.path.join(args.output, "best_lstm.pt"), map_location=device)
    )
    test_loss, test_m = lstm_eval_epoch(model, test_loader, criterion, device)
    print_metrics(test_m, "Test")

    # Full report + confusion matrix
    model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for input_ids, labels in test_loader:
            logits = model(input_ids.to(device))
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels_list.extend(labels.numpy())

    full_report(all_labels_list, all_preds)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model": "BiLSTM" + (" + Attention" if use_attention else " (mean-pool)"),
        "test_metrics": test_m,
        "history": history,
        "args": vars(args),
    }
    results_path = os.path.join(args.output, "lstm_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── (Optional) dump attention weights ─────────────────────────────────────
    if args.save_attention and use_attention:
        attn_path = os.path.join(args.output, "attention_weights.json")
        dump_attention_weights(model, test_loader, vocab, device, attn_path)


if __name__ == "__main__":
    main()