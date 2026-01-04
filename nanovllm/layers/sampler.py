import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(1))
        # shapes: [batch_size, vocab_size]
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max trick for sampling

        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # Step 1: Create empty tensor with same shape as probs
        # noise = torch.empty_like(probs)  # [batch_size, vocab_size]

        # Step 2: Fill with exponential samples (rate=1)
        # noise.exponential_(1)
        # Samples from exponential distribution with rate=1
        # This creates Gumbel noise when combined with log
        # Clamp for numerical stability
        # .clamp_min_(1e-10)
        # Divides probabilities by Gumbel noise
        # This implements: log(probs) - log(-log(U)) where U is uniform
        # Creates Gumbel-max sampling
        return sample_tokens