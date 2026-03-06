import torch
from open_olmo.main import OLMoHybridConfig, OLMoHybrid

if __name__ == "__main__":
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
    print(logits)
    print(logits.shape)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape

    # gen = model.generate(tokens[:, :8], max_new_tokens=16)
    # assert gen.shape[0] == B and gen.shape[1] == 8 + 16, gen.shape

    # print(f"Forward  : {logits.shape}  ✓")
    # print(f"Generate : {gen.shape}  ✓")
