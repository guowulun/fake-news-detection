"""
BERT-based classifier for fake news detection.

Architecture:
  bert-base-uncased (pretrained)
  → [CLS] token representation
  → Dropout(0.1)
  → Linear(768 → 2)

Fine-tuning strategy:
  - Freeze bottom N layers (optional, default: fine-tune all)
  - AdamW optimizer with linear warmup scheduler
  - Gradient clipping (max_norm=1.0)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BERTFakeNewsClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_layers: int = 0,   # freeze bottom N transformer layers
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        # Optionally freeze bottom layers (useful for small datasets)
        if freeze_layers > 0:
            modules = [self.bert.embeddings] + list(
                self.bert.encoder.layer[:freeze_layers]
            )
            for m in modules:
                for param in m.parameters():
                    param.requires_grad = False
            print(f"[BERT] Frozen bottom {freeze_layers} layers.")

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] pooled output
        pooled = outputs.pooler_output          # (B, 768)
        logits = self.classifier(self.dropout(pooled))
        return logits


def count_params(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
