def calc_params(hidden_size, intermediate_size, num_layers, vocab_size, q_lora, kv_lora, heads, rope_dim, v_dim):
    embed = vocab_size * hidden_size
    # FFN: w1(up), w3(gate), w2(down)
    ffn = 3 * hidden_size * intermediate_size
    # MLA Attention:
    # W_DKV (kv_lora)
    w_dkv = hidden_size * kv_lora
    # W_UK (keys) + W_UV (values) -> from kv_lora to (heads * rope_dim) + (heads * v_dim)
    w_uk_uv = kv_lora * (heads * rope_dim + heads * v_dim)
    # W_DQ (q_lora)
    w_dq = hidden_size * q_lora
    # W_UQ (queries) -> from q_lora to (heads * hidden_size/heads) wait, MLA query is heads * q_head_dim
    # standard MLA uses q_head_dim, let's assume it's same as v_dim
    w_uq = q_lora * (heads * rope_dim + heads * v_dim)
    # W_O (output) -> from heads * v_dim to hidden_size
    w_o = (heads * v_dim) * hidden_size
    
    attn = w_dkv + w_uk_uv + w_dq + w_uq + w_o
    layer = ffn + attn
    total = embed + num_layers * layer
    print(f"Embed: {embed/1e6:.1f}M")
    print(f"FFN per layer: {ffn/1e6:.1f}M")
    print(f"Attn per layer: {attn/1e6:.1f}M")
    print(f"Total Params: {total/1e6:.1f}M")

calc_params(
    hidden_size=1536,
    intermediate_size=4096,
    num_layers=24,
    vocab_size=64000,
    q_lora=1024,
    kv_lora=256,
    heads=12,
    rope_dim=64,
    v_dim=128
)
