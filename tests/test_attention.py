import math
import pytest
import torch

from models.lstm_model import AdditiveAttention, FakeNewsLSTM, build_model


# ──────────────────────────────────────────────────────────────────────────────
# AdditiveAttention
# ──────────────────────────────────────────────────────────────────────────────

class TestAdditiveAttention:

    def _make(self, hidden=64):
        return AdditiveAttention(hidden)

    def test_output_shapes(self):
        B, T, H = 4, 12, 64
        attn    = self._make(H)
        seq     = torch.randn(B, T, H)
        ctx, wt = attn(seq)
        assert ctx.shape == (B, H),  f"context shape {ctx.shape}"
        assert wt.shape  == (B, T),  f"weight shape  {wt.shape}"

    def test_weights_sum_to_one(self):
        B, T, H = 3, 10, 64
        attn    = self._make(H)
        seq     = torch.randn(B, T, H)
        _, wt   = attn(seq)
        sums = wt.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"weights don't sum to 1: {sums}"

    def test_mask_zeros_pad_positions(self):
        """PAD positions should receive ~0 attention after masking."""
        B, T, H = 2, 8, 32
        attn    = self._make(H)
        seq     = torch.randn(B, T, H)
        # First sample: only first 5 tokens valid
        mask    = torch.ones(B, T, dtype=torch.bool)
        mask[0, 5:] = False
        _, wt   = attn(seq, mask)
        pad_mass = wt[0, 5:].sum().item()
        assert pad_mass < 1e-4, \
            f"PAD positions have non-zero attention: {pad_mass:.6f}"

    def test_context_is_weighted_sum(self):
        """context = Σ_t α_t · h_t  (verify numerically)."""
        B, T, H = 1, 5, 16
        attn    = self._make(H)
        seq     = torch.randn(B, T, H)
        ctx, wt = attn(seq)
        expected = (wt.unsqueeze(-1) * seq).sum(dim=1)
        assert torch.allclose(ctx, expected, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# FakeNewsLSTM
# ──────────────────────────────────────────────────────────────────────────────

VOCAB  = 500
EMBED  = 50
HIDDEN = 32
LAYERS = 2
B, T   = 8, 24


@pytest.fixture(scope="module")
def model_with_attn():
    return FakeNewsLSTM(
        vocab_size=VOCAB, embed_dim=EMBED, hidden_dim=HIDDEN,
        num_layers=LAYERS, dropout=0.0, freeze_emb=False,
        use_attention=True
    ).eval()


@pytest.fixture(scope="module")
def model_no_attn():
    return FakeNewsLSTM(
        vocab_size=VOCAB, embed_dim=EMBED, hidden_dim=HIDDEN,
        num_layers=LAYERS, dropout=0.0, freeze_emb=False,
        use_attention=False
    ).eval()


class TestFakeNewsLSTM:

    def _ids(self, pad_last=False):
        ids = torch.randint(1, VOCAB, (B, T))
        if pad_last:
            ids[:, -4:] = 0          # last 4 tokens = PAD
        return ids

    # ── shape checks ──────────────────────────────────────────────────────

    def test_logits_shape_with_attention(self, model_with_attn):
        out = model_with_attn(self._ids())
        assert out.shape == (B, 2)

    def test_logits_shape_no_attention(self, model_no_attn):
        out = model_no_attn(self._ids())
        assert out.shape == (B, 2)

    def test_return_attention_shapes(self, model_with_attn):
        ids = self._ids()
        logits, wt = model_with_attn(ids, return_attention=True)
        assert logits.shape == (B, 2)
        assert wt.shape[0]  == B
        assert wt.shape[1]  <= T      # seq after pack/unpad may be shorter

    def test_return_attention_weights_sum_to_one(self, model_with_attn):
        _, wt = model_with_attn(self._ids(), return_attention=True)
        sums  = wt.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    # ── padding handling ──────────────────────────────────────────────────

    def test_padding_does_not_crash(self, model_with_attn):
        ids = self._ids(pad_last=True)
        out = model_with_attn(ids)
        assert out.shape == (B, 2)
        assert not torch.any(torch.isnan(out))

    # ── pretrained embedding loading ──────────────────────────────────────

    def test_load_pretrained_embeddings(self):
        m   = FakeNewsLSTM(VOCAB, EMBED, HIDDEN, freeze_emb=False)
        wt  = torch.randn(VOCAB, EMBED)
        m.load_pretrained_embeddings(wt)
        assert torch.allclose(m.embedding.weight.data, wt)

    def test_load_pretrained_wrong_shape_raises(self):
        m = FakeNewsLSTM(VOCAB, EMBED, HIDDEN)
        with pytest.raises(AssertionError):
            m.load_pretrained_embeddings(torch.randn(VOCAB + 1, EMBED))

    # ── build_model factory ───────────────────────────────────────────────

    def test_build_model_returns_correct_type(self):
        m = build_model(vocab_size=VOCAB, embed_dim=EMBED, hidden_dim=HIDDEN)
        assert isinstance(m, FakeNewsLSTM)

    def test_build_model_no_attention(self):
        m = build_model(vocab_size=VOCAB, embed_dim=EMBED, hidden_dim=HIDDEN,
                        use_attention=False)
        assert not m.use_attention


# ──────────────────────────────────────────────────────────────────────────────
# Gradient flow
# ──────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_gradients_reach_embedding(self):
        """With freeze_emb=False gradients must reach the embedding table."""
        m = FakeNewsLSTM(VOCAB, EMBED, HIDDEN, freeze_emb=False,
                         use_attention=True, dropout=0.0)
        ids    = torch.randint(1, VOCAB, (B, T))
        labels = torch.randint(0, 2, (B,))
        logits = m(ids)
        loss   = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        grad = m.embedding.weight.grad
        assert grad is not None
        assert grad.abs().sum() > 0, "No gradient at embedding layer"

    def test_attention_params_receive_gradients(self):
        m = FakeNewsLSTM(VOCAB, EMBED, HIDDEN, freeze_emb=False,
                         use_attention=True, dropout=0.0)
        ids    = torch.randint(1, VOCAB, (B, T))
        labels = torch.randint(0, 2, (B,))
        logits = m(ids)
        loss   = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        for name, p in m.attention.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"No gradient on attention.{name}"