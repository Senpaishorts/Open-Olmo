# Open-OLMo

<p align="center">
  <a href="https://twitter.com/kyegomezb">
    <picture>
      <source srcset="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
    </picture>
  </a>
  <a href="https://discord.gg/EamjgSaEQf">
    <picture>
      <source srcset="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
    </picture>
  </a>
  <a href="https://pytorch.org/">
    <picture>
      <source srcset="https://img.shields.io/badge/Built%20with-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Built%20with-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    </picture>
  </a>
  <a href="https://github.com/kyegomez/Open-Olmo/stargazers">
    <picture>
      <source srcset="https://img.shields.io/github/stars/kyegomez/Open-Olmo?style=for-the-badge&color=FFD700" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/github/stars/kyegomez/Open-Olmo?style=for-the-badge&color=FFD700" alt="GitHub Stars">
    </picture>
  </a>
  <a href="https://allenai.org/blog/olmohybrid">
    <picture>
      <source srcset="https://img.shields.io/badge/Based%20on-OLMo%20Hybrid-4B9CD3?style=for-the-badge&logo=semanticweb&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Based%20on-OLMo%20Hybrid-4B9CD3?style=for-the-badge&logo=semanticweb&logoColor=white" alt="OLMo Hybrid">
    </picture>
  </a>
</p>

**Unofficial** open-source PyTorch implementation of the OLMo Hybrid architecture introduced by the Allen Institute for AI (Ai2).

> This repository is an independent community re-implementation and is not affiliated with, endorsed by, or officially supported by Ai2.
> For the official release, weights, and tooling, see the [Ai2 OLMo project](https://allenai.org/olmo).

---

## Overview

OLMo Hybrid is a language model architecture that combines two fundamentally different sequence-mixing mechanisms inside a single residual stack:

- **Gated DeltaNet** — a parallelisable linear recurrent neural network (linear-RNN) layer based on the gated delta rule for associative memory.
- **Causal Multi-Head Attention (MHA)** — standard scaled dot-product attention with Rotary Position Embeddings (RoPE).

The two layer types are interleaved in a fixed **3 : 1 ratio** (three DeltaNet layers for every one attention layer). This hybrid design retains the sub-quadratic inference cost of linear RNNs for the bulk of computation while using periodic full attention to prevent information loss in the bounded recurrent state.

---

## Architecture

### Layer Pattern

For a model with `hybrid_ratio = 3` and `num_layers = 8`:

```
Layer 0  →  Gated DeltaNet
Layer 1  →  Gated DeltaNet
Layer 2  →  Gated DeltaNet
Layer 3  →  Multi-Head Attention
Layer 4  →  Gated DeltaNet
Layer 5  →  Gated DeltaNet
Layer 6  →  Gated DeltaNet
Layer 7  →  Multi-Head Attention
```

Each layer consists of:

1. **RMSNorm** pre-normalisation
2. Mixing sublayer (DeltaNet or MHA)
3. Residual addition
4. **RMSNorm** pre-normalisation
5. **SwiGLU** feed-forward network
6. Residual addition

### Gated DeltaNet

The Gated DeltaNet layer maintains an associative-memory matrix state `S` of shape `(B, H, D, D)` updated at every token via the gated delta rule:

```
alpha_t  =  sigmoid(W_alpha * x_t)       in (0,1)^{H x D}   per-element forget gate
beta_t   =  sigmoid(W_beta  * x_t)       in (0,1)^H          delta-rule step size
k_t      =  normalize(W_k * x_t)         in R^{H x D}        key (unit sphere)
v_t      =  W_v * x_t                    in R^{H x D}        value
q_t      =  normalize(W_q * x_t)         in R^{H x D}        query (unit sphere)

S_t      =  (alpha_t  *  S_{t-1})  +  beta_t * (v_t - S_{t-1} k_t) outer k_t
y_t      =  S_t q_t
```

- The `alpha` gate allows the model to selectively forget stale associations.
- The `beta`-scaled delta-rule term writes a corrected association between `k_t` and `v_t` into the memory matrix.
- Normalising keys and queries keeps numerical values bounded regardless of sequence length.
- An additional multiplicative output gate `g = sigmoid(W_g x_t)` is applied to the read-out before the output projection.

The recurrence is available in two forms:

| Mode | Description | Complexity |
|---|---|---|
| `sequential_recurrence` | Token-by-token Python loop; correct by construction | O(T) serial steps |
| `chunked_recurrence` | Block-parallel scan over chunks of size C; intra-chunk work is fully parallelised via triangular matrix multiply | O(T/C) serial steps |

At inference, the DeltaNet state size scales linearly with the number of heads and head dimension — unlike the quadratic KV-cache of full attention.

### Multi-Head Attention

Standard causal multi-head self-attention using `torch.nn.functional.scaled_dot_product_attention`, which dispatches to Flash Attention when available. Rotary Position Embeddings (RoPE) are applied to queries and keys. A single `RotaryEmbedding` instance is shared across all attention layers.

### SwiGLU Feed-Forward Network

```
FFN(x) = dropout( W_down * (SiLU(W_gate * x)  *  W_up * x) )
```

The hidden dimension is set to `round(ffn_mult * d_model)`, rounded up to the nearest multiple of 256 for hardware efficiency.

---

## Repository Structure

```
open_olmo/
    __init__.py
    main.py          # All model components: config, DeltaNet, MHA, FFN, full model
example.py           # Minimal smoke-test: instantiate, forward pass, shape assertion
```

### Key Classes (`open_olmo/main.py`)

| Class / Function | Description |
|---|---|
| `OLMoHybridConfig` | Dataclass holding all hyper-parameters |
| `RotaryEmbedding` | Precomputed RoPE sin/cos tables with lazy rebuild |
| `GatedDeltaNet` | Linear-RNN mixing sublayer with chunked recurrence |
| `MultiHeadAttention` | Causal MHA with RoPE and Flash Attention dispatch |
| `SwiGLUFFN` | Gated feed-forward network |
| `OLMoHybridLayer` | One residual block (mixing sublayer + FFN) |
| `OLMoHybrid` | Full model: embedding, layer stack, output norm, LM head |
| `olmo_hybrid_1b()` | Convenience constructor for ~1 B parameter configuration |
| `olmo_hybrid_7b()` | Convenience constructor for ~7 B parameter configuration |

---

## Installation

```bash
git clone https://github.com/your-org/Open-Olmo.git
cd Open-Olmo
pip install torch
```

No additional dependencies beyond PyTorch are required.

---

## Usage

### Minimal Example

```python
import torch
from open_olmo.main import OLMoHybridConfig, OLMoHybrid

torch.manual_seed(0)

cfg = OLMoHybridConfig(
    vocab_size=1024,
    d_model=256,
    num_heads=4,
    num_layers=8,
    hybrid_ratio=3,
    max_seq_len=512,
    chunk_size=32,
)
model = OLMoHybrid(cfg)

print(f"Layer pattern : {model.layer_types}")
print(f"Parameters    : {model.num_parameters():,}")

B, T = 2, 64
tokens = torch.randint(0, cfg.vocab_size, (B, T))
logits, _ = model(tokens)
# logits: (B, T, vocab_size)
```

### Preset Configurations

```python
from open_olmo.main import olmo_hybrid_1b, olmo_hybrid_7b

model_1b = olmo_hybrid_1b()   # ~1 B parameters
model_7b = olmo_hybrid_7b()   # ~7 B parameters
```

### Autoregressive Generation

```python
generated = model.generate(
    input_ids=tokens[:, :8],
    max_new_tokens=32,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
# generated: (B, 8 + num_generated)
```

The `generate` method prefills recurrent states from the prompt in a single parallel forward pass, then decodes one token per step — reusing the cached DeltaNet states for O(1) per-step inference cost.

### Stateful Inference

```python
logits, states = model(input_ids, return_states=True)

# Pass states to continue inference from where it left off
next_logits, next_states = model(next_token, states=states, return_states=True)
```

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `vocab_size` | 50304 | Vocabulary size (padded to a multiple of 64) |
| `d_model` | 2048 | Residual stream dimension |
| `num_heads` | 16 | Number of attention / DeltaNet heads |
| `num_layers` | 24 | Total number of hybrid layers |
| `ffn_mult` | 8/3 | FFN hidden dim multiplier relative to `d_model` |
| `hybrid_ratio` | 3 | DeltaNet layers per attention layer |
| `max_seq_len` | 8192 | Maximum sequence length for RoPE cache |
| `dropout` | 0.0 | Dropout probability (0 = disabled) |
| `rms_norm_eps` | 1e-5 | Epsilon for RMSNorm |
| `tie_embeddings` | True | Tie input embedding and LM head weights |
| `chunk_size` | 64 | Chunk size for chunked DeltaNet recurrence |
| `init_std` | 0.02 | Weight initialisation standard deviation |
| `rope_base` | 10000.0 | RoPE base frequency |

---

## Citation

If you use this implementation in your research, please cite the original Ai2 work:

```bibtex
@misc{ai2_olmohybrid_2026,
  title        = {Introducing OLMo Hybrid: Combining Transformers and Linear RNNs for Superior Scaling},
  author       = {Ai2},
  year         = {2026},
  howpublished = {\url{https://allenai.org/blog/olmohybrid}},
  note         = {Allen Institute for AI}
}
```

For the delta rule and associative memory foundations underlying GatedDeltaNet, see the relevant prior work on linear recurrent models and the delta rule in sequence modelling.

---

## Disclaimer

This is an **unofficial** community implementation. It reproduces the architecture described in the Ai2 blog post and paper to the best of the authors' understanding. It does not include official pre-trained weights, tokenisers, or training code. For production use, refer to the official Ai2 OLMo repositories.

---

## License

See [LICENSE](LICENSE).
