"""
train_lstm.py  –  BiLSTM + Attention training entry point.

Changes vs. original:
  • Passes use_attention=True to build_model (default, backward-compatible).
  • After evaluation, optionally dumps a JSON of per-sample attention weights
    (--save_attention flag) so you can inspect which tokens drove predictions.
  • Everything else (CLI, data loading, trainer loop) is identical.

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
from torch.utils.data import DataLoader

# ── project imports (unchanged paths) ────────────────────────────────────────
from data.dataset import load_liar, load_fakenewsnet, FakeNewsDataset
from data.vocab   import Vocabulary
from models.lstm_model import build_model
from utils.trainer     import Trainer


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
    p.add_argument("--output",         default="results/lstm")
    # ── new flag ──────────────────────────────────────────────────────────
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
    idx2word = {v: k for k, v in vocab.token2idx.items()}

    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            logits, attn = model(ids, return_attention=True)   # (B,C), (B,T)
            preds  = logits.argmax(dim=-1).cpu().tolist()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    liar_train, liar_val, liar_test = load_liar(args.liar_dir)
    fnn_train,  fnn_val,  fnn_test  = load_fakenewsnet(args.fnn_dir)

    train_raw = liar_train + fnn_train
    val_raw   = liar_val   + fnn_val
    test_raw  = liar_test  + fnn_test

    # ── Vocabulary + GloVe ───────────────────────────────────────────────────
    vocab  = Vocabulary()
    vocab.build([text for text, _ in train_raw])
    glove  = vocab.load_glove(args.glove)           # (vocab_size, embed_dim)

    # ── Datasets / Loaders ───────────────────────────────────────────────────
    mk_ds  = lambda data: FakeNewsDataset(data, vocab, args.max_len)
    train_loader = DataLoader(mk_ds(train_raw), args.batch_size, shuffle=True)
    val_loader   = DataLoader(mk_ds(val_raw),   args.batch_size)
    test_loader  = DataLoader(mk_ds(test_raw),  args.batch_size)

    # ── Model ────────────────────────────────────────────────────────────────
    use_attention = not args.no_attention          # True by default
    model = build_model(
        vocab_size    = len(vocab),
        embed_dim     = glove.shape[1],
        hidden_dim    = args.hidden_dim,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
        use_attention = use_attention,
    ).to(device)
    model.load_pretrained_embeddings(glove)

    print(f"Model  : BiLSTM {'+ Attention' if use_attention else '(mean-pool)'}")
    print(f"Params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, device=device, lr=args.lr, output_dir=args.output)
    trainer.train(train_loader, val_loader, epochs=args.epochs)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = trainer.evaluate(test_loader)
    results_path = os.path.join(args.output, "lstm_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test results → {results_path}")

    # ── (Optional) dump attention weights ────────────────────────────────────
    if args.save_attention and use_attention:
        attn_path = os.path.join(args.output, "attention_weights.json")
        dump_attention_weights(model, test_loader, vocab, device, attn_path)


if __name__ == "__main__":
    main()