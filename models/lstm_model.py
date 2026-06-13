"""
models/lstm_model.py
BiLSTM + Additive Attention classifier for fake-news detection.

Drop-in replacement for the original BiLSTM-only file.
The public API (FakeNewsLSTM, build_model) is unchanged so train_lstm.py
and compare_models.py require zero edits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Attention layer
# ─────────────────────────────────────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive (soft) attention over a sequence of BiLSTM states.

    Given a sequence H ∈ R^{T × 2*hidden}, the layer computes:
        e_t  = v · tanh(W · h_t + b)          (scalar score per step)
        α    = softmax(e)                       (attention weights, T-dim)
        ctx  = Σ_t α_t · h_t                   (context vector, 2*hidden)

    The context vector is returned together with the weights so callers
    can inspect which tokens the model attended to.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # hidden_dim is the *total* BiLSTM output size (2 × lstm_hidden)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        lstm_out: torch.Tensor,          # (batch, seq_len, hidden_dim)
        mask: torch.Tensor | None = None # (batch, seq_len)  True = valid token
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context : (batch, hidden_dim)
        weights : (batch, seq_len)   – attention distribution
        """
        # Score each time step
        scores = self.v(torch.tanh(self.W(lstm_out)))   # (B, T, 1)
        scores = scores.squeeze(-1)                      # (B, T)

        # Mask padding positions with -inf before softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)              # (B, T)

        # Weighted sum → context
        context = torch.bmm(weights.unsqueeze(1), lstm_out)  # (B, 1, H)
        context = context.squeeze(1)                          # (B, H)

        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class FakeNewsLSTM(nn.Module):
    """
    Embedding → BiLSTM (stacked) → Additive Attention → MLP → binary logit.

    Parameters
    ----------
    vocab_size   : vocabulary size
    embed_dim    : GloVe / random embedding dimension        (default 300)
    hidden_dim   : LSTM hidden size *per direction*          (default 128)
    num_layers   : number of stacked BiLSTM layers           (default 2)
    num_classes  : 2 for binary fake/real                    (default 2)
    dropout      : dropout rate applied after embedding
                   and before the classification head        (default 0.3)
    freeze_emb   : freeze pretrained embedding weights       (default True)
    padding_idx  : token index used for PAD                  (default 0)
    use_attention: toggle attention; falls back to mean-pool (default True)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_emb: bool = True,
        padding_idx: int = 0,
        use_attention: bool = True,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.padding_idx   = padding_idx
        bilstm_out_dim     = hidden_dim * 2          # bidirectional

        # ── Embedding ──────────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        if freeze_emb:
            self.embedding.weight.requires_grad_(False)

        self.emb_drop = nn.Dropout(dropout)

        # ── BiLSTM ─────────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Attention ──────────────────────────────────────────────────────
        if use_attention:
            self.attention = AdditiveAttention(bilstm_out_dim)

        # ── Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bilstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,                    # (B, T)
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids       : token indices, shape (batch, seq_len)
        return_attention: if True, also return attention weights

        Returns
        -------
        logits          : (batch, num_classes)
        attn_weights    : (batch, seq_len)  – only when return_attention=True
        """
        # Build padding mask  (True = real token, False = PAD)
        mask = input_ids.ne(self.padding_idx)        # (B, T)

        # Embedding
        x = self.embedding(input_ids)                # (B, T, E)
        x = self.emb_drop(x)

        # BiLSTM – pack to skip PAD steps efficiently
        lengths = mask.sum(dim=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _   = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )                                            # (B, T', 2*H)

        # Pool: attention or mean
        if self.use_attention:
            # Trim mask to match lstm_out length (may differ after padding)
            seq_len = lstm_out.size(1)
            attn_mask = mask[:, :seq_len]
            context, attn_weights = self.attention(lstm_out, attn_mask)
        else:
            # Mean-pool over valid tokens (original behaviour)
            seq_len   = lstm_out.size(1)
            attn_mask = mask[:, :seq_len].unsqueeze(-1).float()
            context   = (lstm_out * attn_mask).sum(1) / attn_mask.sum(1).clamp(min=1)
            attn_weights = None

        logits = self.classifier(context)            # (B, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits

    # ------------------------------------------------------------------
    def load_pretrained_embeddings(self, weight_matrix: torch.Tensor) -> None:
        """
        Load a pretrained embedding matrix (e.g. from GloVe).
        weight_matrix : (vocab_size, embed_dim)
        """
        assert weight_matrix.shape == self.embedding.weight.shape, (
            f"Shape mismatch: expected {self.embedding.weight.shape}, "
            f"got {weight_matrix.shape}"
        )
        self.embedding.weight.data.copy_(weight_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder  (matches original API used by train_lstm.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    vocab_size: int,
    embed_dim: int = 300,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.3,
    freeze_emb: bool = True,
    padding_idx: int = 0,
    use_attention: bool = True,
) -> FakeNewsLSTM:
    """
    Factory function – keeps train_lstm.py import unchanged:

        from models.lstm_model import build_model
        model = build_model(vocab_size=len(vocab), ...)
    """
    return FakeNewsLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        freeze_emb=freeze_emb,
        padding_idx=padding_idx,
        use_attention=use_attention,
    )