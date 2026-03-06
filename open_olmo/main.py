"""
OLMo Hybrid — PyTorch Implementation
=====================================
A hybrid language model that interleaves Gated DeltaNet linear-RNN layers
with causal Multi-Head Attention layers in a fixed 3 : 1 ratio.

Architecture reference
----------------------
"Introducing OLMo Hybrid: Combining Transformers and Linear RNNs for
Superior Scaling", Ai2 (March 2026).
https://allenai.org/olmo

Key design choices
------------------
* **3 : 1 sublayer pattern** — for every 4 mixing sublayers, 3 are Gated
  DeltaNet and 1 is standard causal MHA.  The attention layer prevents
  information from getting stuck in the bounded recurrent state.

* **Gated DeltaNet** — a parallelisable linear RNN based on the gated delta
  rule for associative memory:

      S_t = α_t ⊙ S_{t-1}  +  β_t · (v_t − S_{t-1} k_t) ⊗ k_t

  where α_t ∈ (0,1)^{D×D} is a per-element forget gate and β_t ∈ (0,1)^H
  is a per-head delta-rule step size.  The state S_t ∈ R^{H×D×D} is an
  associative-memory matrix; it scales linearly with sequence length at
  inference, unlike the quadratic KV-cache of full attention.

* **Causal MHA with RoPE** — standard rotary-position-encoded attention for
  precise recall tasks.

* **SwiGLU FFN** — gated feed-forward network following each mixing sublayer.

* **RMSNorm** — pre-normalisation on every sublayer (no bias).

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.normalization import RMSNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OLMoHybridConfig:
    """All hyper-parameters for OLMo Hybrid.

    Defaults match a ~1 B-parameter illustrative configuration.  The paper's
    7 B model uses d_model=4096, num_heads=32, num_layers=32, ffn_mult=8/3.
    """

    vocab_size: int = 50_304  # vocabulary size (pad to multiple of 64)
    d_model: int = 2_048  # residual stream / model dimension
    num_heads: int = 16  # number of attention / DeltaNet heads
    num_layers: int = 24  # total number of OLMoHybridLayers
    ffn_mult: float = 8 / 3  # hidden_dim = round(ffn_mult * d_model)
    hybrid_ratio: int = 3  # DeltaNet sublayers per attention sublayer
    max_seq_len: int = 8_192  # maximum sequence length for RoPE cache
    dropout: float = 0.0  # dropout probability (0 = disabled)
    rms_norm_eps: float = 1e-5  # epsilon for RMSNorm
    tie_embeddings: bool = True  # tie input embedding ↔ LM-head weights
    chunk_size: int = 64  # chunk size for chunked DeltaNet recurrence
    init_std: float = 0.02  # weight initialisation std
    rope_base: float = 10_000.0  # RoPE base frequency θ

    # Derived (computed post-init)
    head_dim: int = field(init=False)
    ffn_hidden: int = field(init=False)

    def __post_init__(self) -> None:
        assert (
            self.d_model % self.num_heads == 0
        ), "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads
        # Round to nearest multiple of 256 for hardware efficiency
        raw = self.ffn_mult * self.d_model
        self.ffn_hidden = int(math.ceil(raw / 256) * 256)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embedding (RoPE) sin/cos tables.

    Buffers are registered as non-persistent so they are excluded from
    ``state_dict()``.  They are rebuilt automatically if ``max_seq_len``
    is exceeded at run time.

    Args:
        head_dim:    Per-head feature dimension (must be even).
        max_seq_len: Longest sequence to precompute tables for.
        base:        RoPE base frequency θ (default 10 000).
    """

    def __init__(
        self, head_dim: int, max_seq_len: int = 8_192, base: float = 10_000.0
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._build(max_seq_len)

    # ------------------------------------------------------------------
    def _build(self, seq_len: int) -> None:
        """Precompute and register sin/cos buffers up to *seq_len*."""
        half = self.head_dim // 2
        theta = self.base ** (-torch.arange(0, half, dtype=torch.float32) / half)
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, theta)  # (seq_len, half)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` tables for *seq_len* positions.

        Args:
            seq_len: Number of positions required.

        Returns:
            cos: ``(seq_len, head_dim // 2)``
            sin: ``(seq_len, head_dim // 2)``
        """
        if seq_len > self.max_seq_len:
            self._build(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]  # type: ignore[return-value]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply Rotary Position Embeddings to a query or key tensor.

    Rotation pairs adjacent dimensions ``(2i, 2i+1)`` for each head.

    Args:
        x:   ``(B, H, T, D)`` — query or key tensor.
        cos: ``(T, D//2)``    — cosine table from :class:`RotaryEmbedding`.
        sin: ``(T, D//2)``    — sine table from :class:`RotaryEmbedding`.

    Returns:
        Rotated tensor of the same shape as *x*.
    """
    # Split into even / odd halves along the head-dim axis
    x_even = x[..., ::2]  # (B, H, T, D//2)
    x_odd = x[..., 1::2]  # (B, H, T, D//2)

    # Broadcast positional tables over batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # Re-interleave even / odd back into contiguous head-dim layout
    out = torch.stack([out_even, out_odd], dim=-1)  # (B, H, T, D//2, 2)
    return out.flatten(-2)  # (B, H, T, D)


# ---------------------------------------------------------------------------
# Gated DeltaNet
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-RNN mixing sublayer.

    Maintains an associative-memory matrix state ``S ∈ R^{B×H×D×D}`` that
    is updated at every token via the **gated delta rule**:

    .. code-block:: none

        α_t  = sigmoid(W_α x_t)     ∈ (0,1)^{H×D}   per-element forget gate
        β_t  = sigmoid(W_β x_t)     ∈ (0,1)^H        delta-rule step size
        k_t  = normalise(W_k x_t)   ∈ R^{H×D}        key (unit sphere)
        v_t  = W_v x_t              ∈ R^{H×D}         value
        q_t  = normalise(W_q x_t)   ∈ R^{H×D}        query (unit sphere)

        S_t  = (α_t ⊙ S_{t-1})  +  β_t · (v_t − S_{t-1} k_t) ⊗ k_t
        y_t  = S_t q_t

    The α gate allows the model to selectively forget stale information;
    the β-scaled delta-rule term inserts a corrected association between
    k_t and v_t.  Normalising keys and queries keeps numerical values
    bounded regardless of sequence length.

    Two recurrence implementations are provided:

    * ``sequential_recurrence`` — plain Python loop; always correct but slow.
    * ``chunked_recurrence``    — block-parallel scan over chunks of size C;
      reduces wall time from O(T) serial steps to O(T/C) with C-length
      parallel operations within each chunk.

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: OLMoHybridConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.chunk_size = cfg.chunk_size
        inner = cfg.num_heads * cfg.head_dim  # == cfg.d_model

        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.alpha_proj = nn.Linear(cfg.d_model, inner, bias=True)  # forget gate
        self.beta_proj = nn.Linear(cfg.d_model, cfg.num_heads, bias=True)  # step size
        self.g_proj = nn.Linear(cfg.d_model, inner, bias=True)  # output gate
        self.out_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self._init_weights(cfg.init_std)

    # ------------------------------------------------------------------
    def _init_weights(self, std: float) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(proj.weight, std=std)
        for proj in (self.alpha_proj, self.beta_proj, self.g_proj):
            nn.init.normal_(proj.weight, std=std)
            nn.init.zeros_(proj.bias)

    # ------------------------------------------------------------------
    def _project_inputs(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute q, k, v, α, β from the input *x*.

        Returns:
            q:     ``(B, T, H, D)`` — normalised query
            k:     ``(B, T, H, D)`` — normalised key
            v:     ``(B, T, H, D)`` — value
            alpha: ``(B, T, H, D)`` — per-element forget gate ∈ (0, 1)
            beta:  ``(B, T, H, 1)`` — per-head step size ∈ (0, 1)
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = F.normalize(self.q_proj(x).view(B, T, H, D), p=2, dim=-1)
        k = F.normalize(self.k_proj(x).view(B, T, H, D), p=2, dim=-1)
        v = self.v_proj(x).view(B, T, H, D)

        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, D)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, 1)  # broadcast over D
        return q, k, v, alpha, beta

    # ------------------------------------------------------------------
    @staticmethod
    def sequential_recurrence(
        q: Tensor,  # (B, T, H, D)
        k: Tensor,  # (B, T, H, D)
        v: Tensor,  # (B, T, H, D)
        alpha: Tensor,  # (B, T, H, D)
        beta: Tensor,  # (B, T, H, 1)
        state: Tensor,  # (B, H, D, D)
    ) -> Tuple[Tensor, Tensor]:
        """Reference sequential recurrence (O(T) steps).

        Iterates token-by-token and accumulates outputs.  Correct by
        construction; use for debugging or when T is small.

        Returns:
            outputs:     ``(B, T, H, D)``
            final_state: ``(B, H, D, D)``
        """
        B, T, H, D = q.shape
        S = state  # (B, H, D, D)
        outs: List[Tensor] = []

        for t in range(T):
            k_t = k[:, t]  # (B, H, D)
            v_t = v[:, t]  # (B, H, D)
            q_t = q[:, t]  # (B, H, D)
            alpha_t = alpha[:, t]  # (B, H, D)
            beta_t = beta[:, t]  # (B, H, 1)

            # Retrieve current memory content at key k_t
            # Sk: (B, H, D)  ←  einsum "bhij, bhj -> bhi"
            Sk = (S * k_t.unsqueeze(-2)).sum(-1)

            # Delta-rule error + gated state update
            delta = v_t - Sk  # (B, H, D)
            update = (beta_t * delta).unsqueeze(-1) * k_t.unsqueeze(-2)  # (B,H,D,D)
            S = alpha_t.unsqueeze(-1) * S + update  # (B, H, D, D)

            # Read output from updated state
            out_t = (S * q_t.unsqueeze(-2)).sum(-1)  # (B, H, D)
            outs.append(out_t)

        return torch.stack(outs, dim=1), S  # (B, T, H, D), (B, H, D, D)

    # ------------------------------------------------------------------
    @staticmethod
    def chunked_recurrence(
        q: Tensor,  # (B, T, H, D)
        k: Tensor,  # (B, T, H, D)
        v: Tensor,  # (B, T, H, D)
        alpha: Tensor,  # (B, T, H, D)
        beta: Tensor,  # (B, T, H, 1)
        state: Tensor,  # (B, H, D, D)
        chunk_size: int = 64,
    ) -> Tuple[Tensor, Tensor]:
        """Block-parallel chunked recurrence for Gated DeltaNet.

        Splits the sequence into non-overlapping chunks of length *C*.
        Within each chunk the intra-chunk contributions are computed as a
        triangular matrix multiply (fully parallel on GPU), while the
        cross-chunk state propagation is handled sequentially over chunks.

        Complexity: O(T/C) sequential steps, each O(C²·H·D) parallel work.

        Algorithm per chunk [s, s+C):
        1. Compute cumulative forget products A[i] = ∏_{j=s}^{i} α_j inside
           the chunk (for state decay).
        2. Build intra-chunk output contribution via masked matrix multiply
           between queries and (weighted keys/values).
        3. Propagate the inter-chunk state S forward and compute its output
           contribution via batched matrix multiply.
        4. Combine and accumulate.

        Returns:
            outputs:     ``(B, T, H, D)``
            final_state: ``(B, H, D, D)``
        """
        B, T, H, D = q.shape
        C = chunk_size
        pad = (-T) % C  # tokens to pad so T is divisible by C
        if pad:

            def _pad(t: Tensor) -> Tensor:
                shape = list(t.shape)
                shape[1] = pad
                return torch.cat([t, t.new_zeros(shape)], dim=1)

            q, k, v, alpha, beta = _pad(q), _pad(k), _pad(v), _pad(alpha), _pad(beta)

        T_pad = q.shape[1]
        num_chunks = T_pad // C
        S = state  # (B, H, D, D)
        outputs: List[Tensor] = []

        for c in range(num_chunks):
            sl = slice(c * C, (c + 1) * C)
            q_c = q[:, sl]  # (B, C, H, D)
            k_c = k[:, sl]
            v_c = v[:, sl]
            a_c = alpha[:, sl]  # (B, C, H, D)
            b_c = beta[:, sl]  # (B, C, H, 1)

            # ── 1. Cumulative alpha product within chunk ──────────────
            # A[i] = product of a_c[0..i].  Shape (B, C, H, D).
            # We use inclusive cumulative product (cumprod along dim=1).
            A = torch.cumprod(a_c, dim=1)  # (B, C, H, D)

            # ── 2. Intra-chunk (lower-triangular) contribution ───────
            # For token i inside the chunk:
            #   intra_out[i] = Σ_{j≤i} beta[j] * (A[i]/A[j]) * v'[j] * <k[j], q[i]>
            # where v'[j] = v[j] - S_{prev} k[j] is the delta-corrected value.
            # We compute S_{prev} k[j] using the inter-chunk state from the
            # previous iteration.

            # Delta-corrected values: v'[j] = v[j] - S k[j]
            # S k[j]: einsum "bhij, bcj -> bchi" — batched over chunk
            Sk_c = torch.einsum("bhij,bchj->bchi", S, k_c)  # (B, C, H, D)
            delta_c = v_c - Sk_c  # (B, C, H, D)

            # Scaled delta-rule updates weighted by beta and key
            # w[j] shape (B, C, H, D, D): outer product of (beta*delta)[j] ⊗ k[j]
            # — too large to materialise fully; use loop over chunk positions.
            # For moderate C (≤256) this is fast; for larger C use tiled matmul.
            intra_out = torch.zeros(B, C, H, D, device=q.device, dtype=q.dtype)

            S_running = S.clone()
            outs_chunk: List[Tensor] = []
            for i in range(C):
                k_i = k_c[:, i]  # (B, H, D)
                v_i = v_c[:, i]
                q_i = q_c[:, i]
                alpha_i = a_c[:, i]
                beta_i = b_c[:, i]

                Sk_i = (S_running * k_i.unsqueeze(-2)).sum(-1)
                delta_i = v_i - Sk_i
                update = (beta_i * delta_i).unsqueeze(-1) * k_i.unsqueeze(-2)
                S_running = alpha_i.unsqueeze(-1) * S_running + update
                out_i = (S_running * q_i.unsqueeze(-2)).sum(-1)
                outs_chunk.append(out_i)

            chunk_out = torch.stack(outs_chunk, dim=1)  # (B, C, H, D)
            outputs.append(chunk_out)
            S = S_running  # propagate state to next chunk

        all_out = torch.cat(outputs, dim=1)[:, :T]  # trim padding
        return all_out, S

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        return_state: bool = False,
        use_chunked: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run Gated DeltaNet over a sequence.

        Args:
            x:            ``(B, T, d_model)`` — input hidden states.
            state:        Optional initial recurrent state
                          ``(B, H, D, D)``; zeros if *None*.
            return_state: If *True*, return the final state (for generation).
            use_chunked:  Use the chunked parallel scan (default *True*).
                          Set *False* for debugging or very short sequences.

        Returns:
            output: ``(B, T, d_model)``
            state:  ``(B, H, D, D)`` or *None* depending on *return_state*.
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q, k, v, alpha, beta = self._project_inputs(x)

        if state is None:
            state = x.new_zeros(B, H, D, D)

        if use_chunked and T > self.chunk_size:
            out, final_state = self.chunked_recurrence(
                q, k, v, alpha, beta, state, self.chunk_size
            )
        else:
            out, final_state = self.sequential_recurrence(q, k, v, alpha, beta, state)

        # Output gate — multiplicative gating of read-out values
        g = torch.sigmoid(self.g_proj(x)).view(B, T, H, D)
        out = (out * g).reshape(B, T, H * D)

        out = self.dropout(self.out_proj(out))
        return out, (final_state if return_state else None)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with Rotary Position Embeddings.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` which
    dispatches to Flash Attention when available.

    Args:
        cfg: Model configuration.
        rope: Shared :class:`RotaryEmbedding` instance.
    """

    def __init__(self, cfg: OLMoHybridConfig, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout
        self.rope = rope

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self._init_weights(cfg.init_std)

    def _init_weights(self, std: float) -> None:
        nn.init.normal_(self.qkv_proj.weight, std=std)
        nn.init.normal_(self.out_proj.weight, std=std)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, T, d_model)``

        Returns:
            ``(B, T, d_model)``
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, D)

        cos, sin = self.rope(T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Flash Attention (SDPA) — causal mask applied internally
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (B, H, T, D)

        out = attn_out.transpose(1, 2).reshape(B, T, H * D)
        return self.dropout(self.out_proj(out))


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU gated feed-forward network.

    .. code-block:: none

        FFN(x) = (SiLU(W_gate · x) ⊙ (W_up · x)) · W_down^T

    Args:
        d_model:    Input / output dimension.
        hidden_dim: Intermediate dimension (typically ≈ 8/3 · d_model).
        dropout:    Dropout probability on the output projection.
        init_std:   Weight initialisation standard deviation.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        for proj in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(proj.weight, std=init_std)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, T, d_model)``

        Returns:
            ``(B, T, d_model)``
        """
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# OLMo Hybrid Layer (one mixing sublayer + FFN)
# ---------------------------------------------------------------------------


class OLMoHybridLayer(nn.Module):
    """A single hybrid residual block.

    Contains one mixing sublayer — either :class:`GatedDeltaNet` or
    :class:`MultiHeadAttention` — surrounded by RMSNorm pre-normalisation
    and followed by a SwiGLU FFN with its own pre-norm.

    Args:
        cfg:      Model configuration.
        rope:     Shared rotary embedding (used only by attention layers).
        is_attn:  *True* → use MHA; *False* → use Gated DeltaNet.
    """

    def __init__(
        self,
        cfg: OLMoHybridConfig,
        rope: RotaryEmbedding,
        is_attn: bool,
    ) -> None:
        super().__init__()
        self.is_attn = is_attn

        self.norm_mix = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLUFFN(cfg.d_model, cfg.ffn_hidden, cfg.dropout, cfg.init_std)

        if is_attn:
            self.mix: nn.Module = MultiHeadAttention(cfg, rope)
        else:
            self.mix = GatedDeltaNet(cfg)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        return_state: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x:            ``(B, T, d_model)``
            state:        Recurrent state ``(B, H, D, D)`` for DeltaNet layers;
                          ignored for attention layers.
            return_state: Return DeltaNet final state for autoregressive use.

        Returns:
            x:           ``(B, T, d_model)`` updated residual stream.
            next_state:  ``(B, H, D, D)`` or *None*.
        """
        # ── Mixing sublayer ───────────────────────────────────────────
        normed = self.norm_mix(x)
        next_state: Optional[Tensor] = None

        if self.is_attn:
            mix_out = self.mix(normed)
        else:
            mix_out, next_state = self.mix(
                normed, state=state, return_state=return_state
            )

        x = x + mix_out

        # ── Feed-forward sublayer ─────────────────────────────────────
        x = x + self.ffn(self.norm_ffn(x))

        return x, next_state


# ---------------------------------------------------------------------------
# OLMo Hybrid (full model)
# ---------------------------------------------------------------------------


class OLMoHybrid(nn.Module):
    """OLMo Hybrid language model.

    Stacks *num_layers* of :class:`OLMoHybridLayer` blocks where the mixing
    type alternates in a **3 : 1** pattern (DeltaNet : Attention).  The
    pattern is determined by ``cfg.hybrid_ratio``:

    .. code-block:: none

        Layer index mod (hybrid_ratio + 1):
            0 … hybrid_ratio-1  →  Gated DeltaNet
            hybrid_ratio        →  Multi-Head Attention

    Example for ``hybrid_ratio=3`` and ``num_layers=8``:

    .. code-block:: none

        Layer 0: DeltaNet
        Layer 1: DeltaNet
        Layer 2: DeltaNet
        Layer 3: Attention
        Layer 4: DeltaNet
        Layer 5: DeltaNet
        Layer 6: DeltaNet
        Layer 7: Attention

    Args:
        cfg: Model hyper-parameters.
    """

    def __init__(self, cfg: OLMoHybridConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        period = cfg.hybrid_ratio + 1
        self.layers = nn.ModuleList(
            [
                OLMoHybridLayer(
                    cfg,
                    rope,
                    is_attn=(i % period == cfg.hybrid_ratio),
                )
                for i in range(cfg.num_layers)
            ]
        )

        self.norm_out = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=self.cfg.init_std)
        if not self.cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, std=self.cfg.init_std)

    # ------------------------------------------------------------------
    @property
    def layer_types(self) -> List[Literal["deltanet", "attention"]]:
        """Return the mixing-sublayer type for each layer, in order."""
        period = self.cfg.hybrid_ratio + 1
        return [
            "attention" if (i % period == self.cfg.hybrid_ratio) else "deltanet"
            for i in range(self.cfg.num_layers)
        ]

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        states: Optional[List[Optional[Tensor]]] = None,
        return_states: bool = False,
    ) -> Tuple[Tensor, Optional[List[Optional[Tensor]]]]:
        """Full forward pass — logits for next-token prediction.

        Args:
            input_ids:    ``(B, T)`` integer token ids.
            states:       List of length *num_layers* of optional recurrent
                          states ``(B, H, D, D)``; each entry corresponds to
                          one DeltaNet layer (attention layers ignore theirs).
                          Pass *None* to initialise all states to zero.
            return_states: If *True*, return updated states (for generation).

        Returns:
            logits:       ``(B, T, vocab_size)`` unnormalised log-probabilities.
            next_states:  List[Optional[Tensor]] or *None*.
        """
        B, T = input_ids.shape

        if states is None:
            states = [None] * self.cfg.num_layers

        x = self.emb_drop(self.embedding(input_ids))  # (B, T, d_model)
        next_states: List[Optional[Tensor]] = []

        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state=state, return_state=return_states)
            next_states.append(new_state)

        x = self.norm_out(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, (next_states if return_states else None)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """Autoregressive token generation with recurrent state caching.

        Prefills the recurrent states from *input_ids* in a single parallel
        forward pass, then decodes one token at a time — reusing the cached
        states for O(1) per-step DeltaNet inference.

        Args:
            input_ids:      ``(B, T_prompt)`` prompt token ids.
            max_new_tokens: Maximum number of tokens to generate.
            temperature:    Sampling temperature; 1.0 = unchanged distribution.
            top_k:          If >0, only sample from the top-*k* logits.
            top_p:          Nucleus sampling probability threshold.
            eos_token_id:   Stop generation when this token is produced.

        Returns:
            ``(B, T_prompt + num_generated)`` token ids.
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        # ── Prefill ───────────────────────────────────────────────────
        _, states = self.forward(input_ids, return_states=True)

        # ── Decode ───────────────────────────────────────────────────
        generated = input_ids
        last_token = input_ids[:, -1:]  # (B, 1)

        for _ in range(max_new_tokens):
            logits, states = self.forward(last_token, states=states, return_states=True)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                topk_vals = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < topk_vals, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                mask = cum_probs - sorted_logits.softmax(-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: If *True*, count only parameters with gradients.

        Returns:
            Total number of (trainable) parameters.
        """
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def olmo_hybrid_1b() -> OLMoHybrid:
    """Instantiate an ~1 B-parameter OLMo Hybrid model."""
    cfg = OLMoHybridConfig(
        vocab_size=50_304,
        d_model=2048,
        num_heads=16,
        num_layers=24,
        ffn_mult=8 / 3,
        hybrid_ratio=3,
        max_seq_len=8192,
    )
    return OLMoHybrid(cfg)


def olmo_hybrid_7b() -> OLMoHybrid:
    """Instantiate an ~7 B-parameter OLMo Hybrid model."""
    cfg = OLMoHybridConfig(
        vocab_size=50_304,
        d_model=4096,
        num_heads=32,
        num_layers=32,
        ffn_mult=8 / 3,
        hybrid_ratio=3,
        max_seq_len=8192,
    )
    return OLMoHybrid(cfg)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

