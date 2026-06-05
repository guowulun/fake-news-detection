# Fake News Detection — LSTM Baseline vs BERT

## Project structure

```
fake_news_detection/
├── data/
│   ├── dataset.py          ← LIAR + FakeNewsNet loaders, PyTorch Dataset wrappers
│   └── vocab.py            ← Vocabulary builder + GloVe loader (for LSTM)
├── models/
│   ├── lstm_model.py       ← BiLSTM classifier
│   └── bert_model.py       ← BERT fine-tune classifier
├── utils/
│   └── trainer.py          ← Train/eval loops, metrics, early stopping
├── train_lstm.py           ← LSTM training entry point
├── train_bert.py           ← BERT training entry point
├── compare_models.py       ← Side-by-side comparison + plots
└── requirements.txt
```

---

## 1. Installation

```bash
pip install -r requirements.txt
```

---

## 2. Download datasets

### LIAR
```bash
wget https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
unzip liar_dataset.zip -d data/liar_dataset
# Files: train.tsv, valid.tsv, test.tsv
```

### FakeNewsNet
```bash
git clone https://github.com/KaiDMML/FakeNewsNet.git data/fakenewsnet
# Or place CSV files manually:
#   data/fakenewsnet/politifact_fake.csv
#   data/fakenewsnet/politifact_real.csv
#   data/fakenewsnet/gossipcop_fake.csv
#   data/fakenewsnet/gossipcop_real.csv
```

### GloVe (for LSTM)
```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/
# Use: data/glove.6B.300d.txt
```

---

## 3. Train LSTM baseline

```bash
python train_lstm.py \
    --liar_dir  data/liar_dataset \
    --fnn_dir   data/fakenewsnet \
    --glove     data/glove.6B.300d.txt \
    --epochs    20 \
    --batch_size 64 \
    --lr        1e-3 \
    --output    results/lstm
```

Key hyperparameters:

| Param | Default | Notes |
|---|---|---|
| `embed_dim` | 300 | GloVe dimension |
| `hidden_dim` | 128 | BiLSTM hidden size (×2 bidirectional) |
| `num_layers` | 2 | BiLSTM depth |
| `dropout` | 0.3 | Applied after embedding + after pooling |
| `freeze_emb` | True | Freeze GloVe weights during training |
| `max_len` | 128 | Token sequence length |

---

## 4. Train BERT (main model)

```bash
python train_bert.py \
    --liar_dir   data/liar_dataset \
    --fnn_dir    data/fakenewsnet \
    --model_name bert-base-uncased \
    --epochs     5 \
    --batch_size 32 \
    --lr         2e-5 \
    --output     results/bert
```

Key hyperparameters:

| Param | Default | Notes |
|---|---|---|
| `model_name` | bert-base-uncased | Can use `bert-large-uncased` or `roberta-base` |
| `max_len` | 128 | BERT sequence length (≤512) |
| `freeze_layers` | 0 | Freeze bottom N transformer blocks (0 = full fine-tune) |
| `dropout` | 0.1 | Before classification head |
| `lr` | 2e-5 | AdamW learning rate |
| `warmup_ratio` | 0.1 | 10% of total steps for linear warmup |
| `weight_decay` | 0.01 | Applied to non-bias / non-LayerNorm params |

---

## 5. Compare results

```bash
python compare_models.py \
    --lstm_results results/lstm/lstm_results.json \
    --bert_results results/bert/bert_results.json \
    --output       results/comparison
```

Outputs `comparison_summary.json` and `training_curves.png`.

---

## 6. Expected results (reference)

| Model | Accuracy | F1-macro | Notes |
|---|---|---|---|
| BiLSTM | ~78–83% | ~0.77–0.82 | With GloVe 300d, frozen |
| BERT-base | ~88–92% | ~0.87–0.91 | Full fine-tune, 5 epochs |

Exact numbers vary by dataset split and hardware.

---

## 7. Design decisions

**Binary labels**: LIAR's 6-class scheme (true → pants-fire) is collapsed to
real (1) and fake (0) using the mid-point: {true, mostly-true, half-true} = real,
{barely-true, false, pants-fire} = fake. This matches FakeNewsNet's binary scheme
and simplifies cross-dataset comparison.

**BERT pooling**: We use BERT's `pooler_output` ([CLS] token passed through a
dense + tanh layer), which is the standard approach for sentence classification.
An alternative is mean-pooling all token states — worth ablating.

**Max-length**: 128 tokens covers >95% of LIAR statements and most FakeNewsNet
headlines. Increase to 256 or 512 for article-level inputs (higher VRAM cost).

**Optimizer**: LSTM uses Adam with ReduceLROnPlateau. BERT uses AdamW with
linear warmup — important for stable transformer fine-tuning.
