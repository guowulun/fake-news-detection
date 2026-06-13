# Fake News Detection — LSTM Baseline vs BERT

## Project structure

```
fake_news_detection/
├── data/
│   ├── dataset.py          ← LIAR + FakeNewsNet loaders, PyTorch Dataset wrappers
│   └── vocab.py            ← Vocabulary builder + GloVe loader (for LSTM)
├── models/
│   ├── lstm_model.py       ← BiLSTM + Attention
│   └── bert_model.py       ← BERT fine-tune classifier
├── utils/
│   └── trainer.py          ← Train/eval loops, metrics, early stopping
├── train_lstm.py           ← LSTM training entry point
├── train_bert.py           ← BERT training entry point
├── compare_models.py       ← Side-by-side comparison + plots
└── requirements.txt
```

