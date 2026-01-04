"""RMSNorm (Root Mean Square Layer Normalization), a variant of layer normalization used in models like LLaMA. 
It normalizes using RMS instead of mean and variance."""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS(x) = sqrt(mean(x²))
    x_norm = x / (RMS(x) + eps)
    output = weight * x_norm"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        #Normalization can reduce gradient magnitudes
        # Weight multiplication helps restore signal strength
        # Improves training stability
        self.weight = nn.Parameter(torch.ones(hidden_size))

    
    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float() # Promote to float32 for numerical stability
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # torch.rsqrt: Reciprocal square root (1/sqrt)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight) 
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        # x: Normalized and scaled output
        # residual: Pre-normalization value (for next layer or gradient flow)
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)