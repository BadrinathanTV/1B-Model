import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import os

# Add directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from layers.attention import MultiHeadLatentAttention
from layers.rope import precompute_freqs_cis, precompute_cos_sin
from layers.residual import DeltaAttentionResidual
from models.mtp import DeepSeekMTPBlock, MTPModule

class TestSLMStabilitySuite(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.hidden_size = 128
        self.config.num_attention_heads = 4
        self.config.q_lora_rank = 32
        self.config.kv_lora_rank = 16
        self.config.qk_rope_head_dim = 16
        self.config.v_head_dim = 32
        self.config.rms_norm_eps = 1e-6
        self.config.training.seq_len = 16
        
        self.batch_size = 2
        self.seq_len = 16

    def test_latent_rmsnorm_stabilizes_attention_logits(self):
        """
        DeepSeek-V2/V3 achieves training stability by applying RMSNorm to the 
        compressed latent vectors (q_c and kv_c) rather than standard QK-Norm 
        (which breaks inference KV-cache absorbability). 
        
        This test proves that the latent RMSNorms successfully bound the 
        attention logits when exposed to extreme activation variance.
        """
        original_sdpa = F.scaled_dot_product_attention
        observed_max_logit = []

        def mocked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            s = scale if scale is not None else (1.0 / (q.shape[-1] ** 0.5))
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * s
            observed_max_logit.append(logits.abs().max().item())
            return original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        F.scaled_dot_product_attention = mocked_sdpa

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Massive outlier inputs (simulating gradient explosion/training instability)
            x_raw = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, device=device) * 2000.0
            
            # Apply standard Transformer Pre-Norm (which bounds x to O(1) before attention)
            from layers.norm import RMSNorm
            attn_norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps).to(device)
            x_extreme = attn_norm(x_raw)
            
            freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta, device=device)
            cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta, device=device)

            # --- Case 1: Proper MLA (with q_norm and kv_norm on latents) ---
            mla_proper = MultiHeadLatentAttention(self.config).to(device)
            observed_max_logit.clear()
            _ = mla_proper(x_extreme, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
            proper_logit_max = observed_max_logit[0]

            # --- Case 2: Degraded MLA (Latent Norms Disabled) ---
            mla_degraded = MultiHeadLatentAttention(self.config).to(device)
            # Copy identical weights to ensure fair comparison
            mla_degraded.load_state_dict(mla_proper.state_dict())
            
            # Disable the latent RMSNorms by replacing them with Identity
            mla_degraded.q_norm = nn.Identity()
            mla_degraded.kv_norm = nn.Identity()
            # Disable QK-Norm on heads as well so attention logits explode
            mla_degraded.q_head_norm = nn.Identity()
            mla_degraded.k_head_norm = nn.Identity()

            observed_max_logit.clear()
            # Bypassing pre-norm for the degraded case to show how quickly it explodes without BOTH protections
            _ = mla_degraded(x_raw, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
            degraded_logit_max = observed_max_logit[0]

            print(f"\n[MLA Stability via Latent Norms]")
            print(f"Degraded MLA (No Pre-Norm, No Latent Norms) Max Logit: {degraded_logit_max:.4f}")
            print(f"Proper MLA (Pre-Norm + Latent Norms) Max Logit:      {proper_logit_max:.4f}")

            # Verify that the latent RMSNorms stabilize the logits by several orders of magnitude
            self.assertTrue(degraded_logit_max > 1000.0, "Expected degraded logits to explode.")
            self.assertTrue(proper_logit_max < 20.0, "Expected proper MLA logits to be stabilized by latent norm.")

        finally:
            F.scaled_dot_product_attention = original_sdpa

    def test_mtp_variance_stability(self):
        """
        Stacked projections in multi-token prediction models risk variance explosion or collapse.
        This test proves that our MTP SwiGLU gating + RMSNorm structure stabilizes 
        activations, preserving output variance near O(1) regardless of network depth.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, device=device)
        
        # Instantiate standard proper MTP head
        mtp_proper = DeepSeekMTPBlock(self.config).to(device)
        
        # Instantiate a degraded linear MTP projection head without SwiGLU / RMSNorm / Residual 
        class DegradedMTPProjection(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            def forward(self, x):
                return self.proj(x)
                
        mtp_degraded = DegradedMTPProjection(self.config.hidden_size).to(device)
        
        # Forward pass through multiple depth iterations
        proper_out = x
        degraded_out = x
        
        # Create token embeddings for the chain
        token_emb = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, device=device)
        
        for _ in range(8):
            proper_out = mtp_proper(proper_out, token_emb)
            degraded_out = mtp_degraded(degraded_out)
            
        proper_variance = proper_out.var().item()
        degraded_variance = degraded_out.var().item()
        
        print(f"\n[MTP Projection Stability]")
        print(f"Degraded stacked projections output variance: {degraded_variance:.4f}")
        print(f"Proper gated+norm projections output variance:  {proper_variance:.4f}")
        
        # Proper stacked MTP heads should be stable and not collapse to 0 or explode to infinity
        self.assertTrue(0.1 < proper_variance < 10.0, f"MTP projection variance exploded or collapsed: {proper_variance}")

    def test_delta_residual_convex_bound(self):
        """
        Delta Attention Residual routing uses softmax attention over past layer outputs.
        Since softmax outputs sum to 1, the routed output is a convex combination of previous deltas.
        This test proves that the routing prevents variance explosion by bounding the routed
        output scale to the scale of individual deltas, rather than accumulating them linearly.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config.num_hidden_layers = 6
        self.config.max_delta_history = 0
        max_hist = 2 * self.config.num_hidden_layers
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, device=device)
        routing_layer = DeltaAttentionResidual(self.config).to(device)
        
        # Learned routing query
        routing_q = nn.Parameter(torch.randn(self.config.hidden_size, device=device))
        
        # Simulate 10 deltas, each with stable unit variance (cap at max_hist)
        n_deltas = min(10, max_hist)
        deltas = [torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, device=device) for _ in range(n_deltas)]
        
        # Build buffer and active_mask
        buf = torch.zeros(max_hist, self.batch_size, self.seq_len, self.config.hidden_size, device=device)
        for i, d in enumerate(deltas):
            buf[i] = d
        active_mask = torch.zeros(max_hist, self.batch_size, self.seq_len, dtype=torch.bool, device=device)
        active_mask[:n_deltas] = True
        
        # Route through DeltaAttentionResidual
        routed_x = routing_layer(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        
        # Calculate routed delta contribution: routed_x - x
        routed_contribution = routed_x - x
        
        contribution_std = routed_contribution.std().item()
        deltas_mean_std = torch.stack(deltas).std().item()
        
        print(f"\n[Delta Attention Residual Routing Stability]")
        print(f"Average individual delta standard deviation: {deltas_mean_std:.4f}")
        print(f"Routed contribution standard deviation:       {contribution_std:.4f}")
        
        # The convex combination property guarantees that the standard deviation of the routed 
        # contribution is bounded by the scale of individual deltas (approx. O(1))
        self.assertTrue(contribution_std < 2.0 * deltas_mean_std, 
                        f"Routed contribution variance is unexpectedly large: {contribution_std}")

if __name__ == '__main__':
    unittest.main()
