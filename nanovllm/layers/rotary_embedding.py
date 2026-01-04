from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:

    # [y1]   [cos(θ)  -sin(θ)] [x1]
    # [y2] = [sin(θ)   cos(θ)] [x2]
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # Ensures rotary dimension equals head size (full rotation).
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # Step 1: Create indices for pairs
        # indices = torch.arange(0, rotary_dim, 2)  # [0, 2, 4, 6, ...] (every other number)
        # Example: rotary_dim=128 → [0, 2, 4, ..., 126] (64 values)

        # Step 2: Normalize by rotary_dim
        # normalized = indices / rotary_dim  # [0/128, 2/128, 4/128, ...]
        # Example: [0.0, 0.0156, 0.0312, ..., 0.9844]

        # Step 3: Compute base^normalized
        # powers = base ** normalized
        # Example with base=10000:
        # [10000^0, 10000^0.0156, 10000^0.0312, ...]
        # = [1.0, 1.36, 1.85, ...]

        # Step 4: Take inverse (1 / value)
        # inv_freq = 1.0 / powers
        # Example: [1.0, 0.735, 0.541, ...]

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # What einsum does:
        # "i,j -> ij": Outer product
        # For each position i and frequency j, compute t[i] * inv_freq[j]

        # ------------------------------------------------------------

        # t = [0, 1, 2]           # Positions
        # inv_freq = [1.0, 0.5]    # Two frequency pairs

        # freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # freqs[0, 0] = 0 * 1.0 = 0.0
        # freqs[0, 1] = 0 * 0.5 = 0.0
        # freqs[1, 0] = 1 * 1.0 = 1.0
        # freqs[1, 1] = 1 * 0.5 = 0.5
        # freqs[2, 0] = 2 * 1.0 = 2.0
        # freqs[2, 1] = 2 * 0.5 = 1.0

        # Result:
        # freqs = [[0.0, 0.0],
        #           [1.0, 0.5],
        #           [2.0, 1.0]]
        # Shape: [max_position_embeddings, num_pairs]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # Step 1: Concatenate cos and sin
        # cos shape: [max_pos, num_pairs]
        # sin shape: [max_pos, num_pairs]
        # cos_sin = torch.cat((cos, sin), dim=-1)
        # Shape: [max_pos, 2 * num_pairs]
        # Layout: [cos_pair0, cos_pair1, ..., sin_pair0, sin_pair1, ...]

        # Step 2: Add dimension
        # cache = cos_sin.unsqueeze_(1)
        # Shape: [max_pos, 1, 2 * num_pairs]
        # The middle dimension is for broadcasting with batch/head dimensions
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb