"""
Transformer Components
=======================

TransformerBlock and SLMModel with:
  - Pre-Norm + Post-Norm (attention & FFN sublayers)
  - QK-Norm (in attention module)
  - Truncated normal weight initialization (DeepSeek-V3 style)
  - Gradient checkpointing
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import SLMConfig
from layers.norm import RMSNorm
from layers.attention import MultiHeadLatentAttention
from layers.ffn import DenseFFN
from layers.residual import DeltaAttentionResidual
from models.embedding import Embedding


class TransformerBlock(nn.Module):
    """Transformer block with Delta Attention Residuals and standard BF16 precision.

    Uses both Pre-Norm (before sublayer) and Post-Norm (after sublayer) for
    tighter activation magnitude control ("Spike No More", extended norm studies).
    """

    def __init__(self, config: SLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.max_delta_history = config.max_delta_history
        # Resolve max_delta_history=0 ("full history") to a concrete buffer size.
        # Each layer produces 2 deltas (attn + ffn), so the physical max is 2*num_layers.
        if self.max_delta_history == 0:
            self.max_delta_history = 2 * config.num_hidden_layers

        # Pre-norms (standard)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Post-norms (controls output magnitude of sublayers)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.attn = MultiHeadLatentAttention(config)
        self.ffn = DenseFFN(config)

        # Delta Attention Residual routing block
        self.delta_residual = DeltaAttentionResidual(config)

        # Learned query parameters (zero-initialized as per paper)
        self.routing_q_attn = nn.Parameter(torch.zeros(config.hidden_size))
        self.routing_q_ffn = nn.Parameter(torch.zeros(config.hidden_size))



    def forward(self, x, past_deltas=None, delta_buffer=None, active_mask=None, current_idx=None, 
                freqs_cis=None, cos_cache=None, sin_cache=None, past_key_value=None):
        
        # ---------------------------------------------------------
        # 1. ATTENTION SUBLAYER
        # ---------------------------------------------------------
        h_attn = self.delta_residual(
            x, self.routing_q_attn, 
            past_deltas=past_deltas, 
            delta_buffer=delta_buffer, 
            active_mask=active_mask
        )
        
        attn_out = self.attn(
            self.attn_norm(h_attn), freqs_cis=freqs_cis,
            cos_cache=cos_cache, sin_cache=sin_cache,
            past_key_value=past_key_value
        )
        if past_key_value is not None:
            v_attn, present_key_value = attn_out
        else:
            v_attn = attn_out
            present_key_value = None
            
        v_attn = self.post_attn_norm(v_attn)
        x = x + v_attn

        # Update static buffer for generation
        if delta_buffer is not None:
            delta_buffer, active_mask, current_idx = self.delta_residual.update_state(
                v_attn, delta_buffer, active_mask, current_idx
            )

        # ---------------------------------------------------------
        # 2. FFN SUBLAYER
        # ---------------------------------------------------------
        # If using lists (training), the FFN needs access to v_attn!
        if past_deltas is not None:
            current_past_deltas = past_deltas + [v_attn]
            if len(current_past_deltas) > self.max_delta_history:
                current_past_deltas = current_past_deltas[-self.max_delta_history:]
        else:
            current_past_deltas = None

        h_ffn = self.delta_residual(
            x, self.routing_q_ffn, 
            past_deltas=current_past_deltas, 
            delta_buffer=delta_buffer, 
            active_mask=active_mask
        )
        
        v_ffn = self.post_ffn_norm(self.ffn(self.ffn_norm(h_ffn)))
        x = x + v_ffn

        # Update static buffer for generation
        if delta_buffer is not None:
            delta_buffer, active_mask, current_idx = self.delta_residual.update_state(
                v_ffn, delta_buffer, active_mask, current_idx
            )

        # Return the new deltas so the model loop can append them safely
        if past_key_value is not None:
            return x, v_attn, v_ffn, delta_buffer, active_mask, current_idx, present_key_value
        return x, v_attn, v_ffn, delta_buffer, active_mask, current_idx


class SLMModel(nn.Module):
    """The full 1B SLM integrating Delta Residuals, MLA, Dense FFN, and MTP.

    Stability features:
      - Truncated normal initialization (sigma=config.init_std)
      - Residual sublayer output scaling by 1/sqrt(2*num_layers)
      - Post-attention/FFN normalization
      - QK-Norm (inside MLA)
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.embed = Embedding(config)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.word_embeddings.weight

        # Output logit scaling / temperature
        base_scale = config.output_logit_scale

        if config.output_logit_scale_trainable:
            self.logit_scale = nn.Parameter(torch.tensor(base_scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(base_scale))

        # Apply weight initialization AFTER all modules are created
        self.apply(self._init_weights)

        # MTP module (uses shared embedding + lm_head)
        from models.mtp import MTPModule
        self.mtp = MTPModule(config, self.lm_head, shared_emb=self.embed)

    def _init_weights(self, module):
        """DeepSeek-V3 style truncated normal initialization.

        Also scales residual sublayer outputs (o_proj, down_proj) by
        1/sqrt(2*num_layers) to keep residual stream magnitude stable
        across depth (GPT-2 / "Spike No More" recommendation).
        """
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)

        # Scale residual output projections for depth stability
        residual_scale = 1.0 / math.sqrt(2.0 * self.config.num_hidden_layers)
        if isinstance(module, MultiHeadLatentAttention):
            with torch.no_grad():
                module.o_proj.weight.mul_(residual_scale)
        elif isinstance(module, DenseFFN):
            with torch.no_grad():
                module.down_proj.weight.mul_(residual_scale)

    def forward(self, input_ids, freqs_cis=None, cos_cache=None, sin_cache=None,
                return_hidden_states: bool = False, use_mtp: bool = True,
                target_ids=None, use_cache: bool = False, past_deltas=None,
                delta_buffer=None, active_mask=None, current_idx=None,
                past_key_values=None):
        
        x = self.embed(input_ids)
        batch_size, seq_len = input_ids.shape

        max_deltas = self.config.max_delta_history
        if max_deltas == 0:
            max_deltas = 2 * self.config.num_hidden_layers

        # --- Phase Routing: Lists for Training, Buffers for Generation ---
        if self.training or not use_cache:
            if past_deltas is None:
                past_deltas = []
        else:
            if delta_buffer is None:
                delta_buffer, active_mask, current_idx = self.layers[0].delta_residual.init_state(
                    batch_size=batch_size, seq_len=seq_len, max_deltas=max_deltas, 
                    dtype=x.dtype, device=x.device
                )

        interval = getattr(self.config, "gradient_checkpointing_interval", 1)

        for i, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.training and self.config.gradient_checkpointing and (i % interval == 0):
                # Gradient checkpointing requires all inputs/outputs to be tensors. 
                # We pass the list in, but rely on the returned v_attn/v_ffn to update it.
                out = checkpoint(
                    layer, x, past_deltas, delta_buffer, active_mask, current_idx,
                    freqs_cis, cos_cache, sin_cache, layer_past_key_value,
                    use_reentrant=False
                )
            else:
                out = layer(
                    x, past_deltas, delta_buffer, active_mask, current_idx,
                    freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache,
                    past_key_value=layer_past_key_value
                )
            
            if layer_past_key_value is not None:
                x, v_attn, v_ffn, delta_buffer, active_mask, current_idx, present_key_value = out
                past_key_values[i] = present_key_value
            else:
                x, v_attn, v_ffn, delta_buffer, active_mask, current_idx = out
            
            # Safely accumulate history during training outside the checkpoint block
            if past_deltas is not None:
                past_deltas = past_deltas + [v_attn, v_ffn]
                if len(past_deltas) > max_deltas:
                    past_deltas = past_deltas[-max_deltas:]

        x = self.norm(x)

        if return_hidden_states:
            return self.mtp.forward_hidden(x, use_mtp=use_mtp, logit_scale=self.logit_scale, target_ids=target_ids)

        logits_list = self.mtp(x, use_mtp=use_mtp, logit_scale=self.logit_scale)
        if past_key_values is not None:
            return logits_list, past_key_values
        return logits_list

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9):
        """Autoregressive generation handling both prefill and decode phases."""
        self.eval()
        
        # 1. Initialize the KV Cache structure (Time-wise state)
        # MLA typically requires a cache object or list of tensors per layer
        past_key_values = [None] * self.config.num_hidden_layers
        
        # RoPE Caches (assuming you have a precomputed cache accessible)
        # freqs_cis = self.freqs_cis[:seq_len_plus_max_new_tokens]
        # For simplicity, we assume freqs_cis, cos_cache, sin_cache are managed dynamically or externally
        # in a real generation loop, or we can just omit them here if handled automatically by the model.
        
        for step in range(max_new_tokens):
            # ---------------------------------------------------------
            # PREFILL PHASE (First step) vs DECODE PHASE (Subsequent steps)
            # ---------------------------------------------------------
            is_prefill = (step == 0)
            
            # During prefill, we process the whole prompt.
            # During decode, we only process the single newly generated token.
            current_input_ids = input_ids if is_prefill else next_token
            
            # ---------------------------------------------------------
            # FORWARD PASS
            # ---------------------------------------------------------
            # Phase Routing magic happens here:
            # - prefill: use_cache=False -> uses fast Python lists for depth-routing
            # - decode:  use_cache=True  -> allocates static delta buffer for compilation
            out = self.forward(
                input_ids=current_input_ids,
                use_cache=not is_prefill, 
                past_key_values=past_key_values, 
            )
            
            # Unpack
            logits = out[0] if isinstance(out, tuple) else out
            if isinstance(out, tuple):
                past_key_values = out[1]
            
            # We only care about the logits for the very last token
            # Note: MTP returns a list of logits (one per depth). We use depth 0 for the base model token.
            base_logits = logits[0] if isinstance(logits, list) else logits
            next_token_logits = base_logits[:, -1, :]
            
            # ---------------------------------------------------------
            # SAMPLING
            # ---------------------------------------------------------
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                
                # Top-K filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the predicted token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop generation if EOS token is reached (assuming EOS id is 2)
            if (next_token == 2).any():
                break
                
        return input_ids
