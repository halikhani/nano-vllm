from re import L
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        # This is a base class. A subclass would typically:
        # Implement weight_loader to handle sharded loading
        # Override forward() to handle tensor-parallel computation
        # Potentially adjust weight shapes based on tp_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    ''' Standard linear layer with no tensor parallelism. Each rank has a full copy. '''
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        ''' Direct copy of full weights to all ranks '''
        param.data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Standard linear transformation: x @ weight.T + bias '''
        return F.linear(x, self.weight, self.bias)
    

class ColumnParallelLinear(LinearBase):
    ''' Splits the output dimension across ranks. Each rank computes part of the output. '''
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # Each rank gets output_size // tp_size rows of the weight matrix
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0) # 0 for divide over output, 1 for divide over input

        # Full weight: [3072, 768]  (output_size=3072, input_size=768)
        # tp_size = 4

        # Rank 0: weight[0:768, :]     (first 768 rows)
        # Rank 1: weight[768:1536, :]  (next 768 rows)
        # Rank 2: weight[1536:2304, :]
        # Rank 3: weight[2304:3072, :]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim) # Each rank's self.weight shape: [768, 768] (not [3072, 768])
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    ''' Merges multiple column-parallel linear layers (e.g., Q, K, V) into one weight matrix. 
    Computes all projections in a single matrix multiplication. '''
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int], # List of sizes for each merged layer
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # loaded_shard_id: which logical projection this is (e.g., 0 for Q, 1 for K, 2 for V)
        param_data = param.data
        # Step 1: Compute where this projection lives inside the merged tensor
        # This tells you where inside param.data this projection starts.
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # Step 2: Compute how many rows this rank owns for this projection
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # Step 3: Narrow the destination tensor
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        # equivalent to: loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]
        param_data.copy_(loaded_weight)
    
    
# Visual illustration (QKV example, tp_size = 2)
# Full merged weight (conceptual)
# | Q rows | K rows | V rows |

# Rank 0 local parameter
# | Q0 | K0 | V0 |

# Rank 1 local parameter
# | Q1 | K1 | V1 |



