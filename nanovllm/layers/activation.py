import torch
from torch import nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    # what is torch.compile ?  
    # @torch.compile is a PyTorch 2.x decorator that tells PyTorch to trace, optimize, and JIT-compile a function 
    # (usually a model’s forward pass) into a faster executable graph using TorchDynamo + AOTAutograd + backend compilers 
    # (Inductor by default).
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Splits the last dimension into two equal parts
        # No data copy (view-based when possible)
        x, y = x.chunk(2, dim=-1) 
        # SwiGLU = Swish + GLU
        # SiLU(x) = x * σ(x)
        return F.silu(x) * y
