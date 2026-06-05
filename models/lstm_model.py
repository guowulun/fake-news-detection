"""
LSTM Baseline model for fake news detection.

Architecture:
  Embedding (GloVe 300d, frozen or fine-tuned)
  → BiLSTM (hidden=128, layers=2)
  → Max-over-time pooling
  → Dropout(0.3)
  → Linear(256 → 2)
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings=None,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Classifier head: bidirectional → 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len) long tensor
        Returns:
            logits: (batch, num_classes)
        """
        embedded = self.dropout(self.embedding(input_ids))   # (B, L, E)
        outputs, _ = self.lstm(embedded)                      # (B, L, 2H)

        # Max-over-time pooling
        pooled, _ = outputs.max(dim=1)                        # (B, 2H)

        logits = self.classifier(self.dropout(pooled))
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
