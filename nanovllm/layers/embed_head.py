''' Two classes for tensor-parallel (TP) distributed training:
VocabParallelEmbedding — vocabulary-parallel embedding layer
ParallelLMHead — parallel language model head (extends the embedding class) '''
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils import context

class VocabParallelEmbedding(nn.Module):
    ''' Distributes the vocabulary embedding table across multiple GPUs/ranks in tensor-parallel training.
    Each rank holds a shard of the vocabulary. '''
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_dize = dist.get_world_size()
        # Each rank holds num_embeddings // tp_size embeddings
        assert num_embeddings % self.tp_size == 0, "num_embeddings must be divisible by tp_size"
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # Each rank stores only (num_embeddings_per_partition, embedding_dim) instead of (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        ''' Custom loader for checkpoint loading
        Extracts the rank’s shard from the full weight tensor '''
        param_data = param.data
        shared_size = param.data.size(0)
        start_idx = self.tp_rank * shared_size
        # tensor.narrow(dim, start, length)
        #     Returns a view of the tensor that:
        #     slices along dimension dim
        #     starting at index start
        #     taking exactly length elements
        loaded_weight = loaded_weight.narrow(0, start_idx, shared_size)
        # param_data → local shard tensor
        # loaded_weight → view into the checkpoint tensor
        # copy_ performs an in-place copy:
        param_data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx) # converts global indices to local indices
        # x: Contains local indices (and zeros for out-of-shard tokens).
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # mask.unsqueeze(1) * y: Element-wise multiplication to zero out embeddings for tokens not in this rank’s shard.
            y = mask.unsqueeze(1) * y 
            dist.all_reduce(y)
            # Rank 0: y = [[emb_0], [0], [emb_2], [0]]]
            # Rank 1: y = [[0], [emb_1], [0], [emb_3]]]
            # After all_reduce: Both ranks get [[emb_0], [emb_1], [emb_2], [emb_3]]]

            # default for all_reduce is ReduceOp.SUM
        return y
    

class ParallelLMHead(VocabParallelEmbedding):
    pass