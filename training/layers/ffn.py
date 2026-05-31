import torch.nn as nn
import torch.nn.functional as F

from config import SLMConfig

# Winner: Liger Kernel SwiGLU (4.4x faster than eager, has autograd)
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    LIGER_SWIGLU = True
except ImportError:
    LIGER_SWIGLU = False


class DenseFFN(nn.Module):
    """Standard Dense Feed-Forward Network with Liger fused SwiGLU kernel."""
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        if LIGER_SWIGLU and x.is_cuda:
            hidden = LigerSiLUMulFunction.apply(gate, up)
        else:
            hidden = F.silu(gate) * up
            
        return self.down_proj(hidden)
