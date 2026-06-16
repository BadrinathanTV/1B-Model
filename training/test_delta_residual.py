import torch
from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis

def test_model():
    print("Testing SLMModel with Delta Block...")
    
    # Tiny config for fast testing
    config = SLMConfig(
        vocab_size=1000,
        hidden_size=64, # small hidden size
        num_hidden_layers=4, # small number of layers
        num_attention_heads=2,
        intermediate_size=128,
        delta_block_size=2,
        max_delta_history=2,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SLMModel(config).to(device)
    model.train()
    
    bsz = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len)).to(device)
    
    # Precompute freqs_cis for training
    freqs_cis = precompute_freqs_cis(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=device
    )
    
    print("Running training forward pass (list mode)...")
    try:
        out = model(input_ids, freqs_cis=freqs_cis, use_cache=False)
        if isinstance(out, list):
            print(f"Forward success! MTP returned {len(out)} logit tensors.")
        else:
            print(f"Forward success! Logits shape: {out.shape}")
    except Exception as e:
        print(f"Training forward failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running generation forward pass (buffer mode)...")
    try:
        model.eval()
        with torch.no_grad():
            # Generate 4 new tokens
            out_gen = model.generate(input_ids[:, :4], max_new_tokens=4)
        print(f"Generation success! Output shape: {out_gen.shape}")
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("All tests passed!")

if __name__ == "__main__":
    test_model()
