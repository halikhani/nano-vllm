import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # Maximum tokens in a batch (prefill + decode). Limits batch size.
    max_num_seqs: int = 512 # maximum sequences processed in parallel in a batch
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1 # number of GPUs to use for tensor parallelism
    enforce_eager: bool = False # if true, disables cuda graph optimizations
    hf_config: AutoConfig | None = None # HuggingFace config to use for the model
    eos: int = -1 # end of sentence token id
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1 # Total number of KV cache blocks (calculated from GPU memory if -1).

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len # Ensures the batch token limit can fit at least one full-length sequence.


# How This Fits in the System
# LLMEngine creates a Config from user arguments.
# The config is passed to Scheduler, ModelRunner, and BlockManager.
# BlockManager uses num_kvcache_blocks and kvcache_block_size to allocate memory.
# Scheduler uses max_num_batched_tokens and max_num_seqs to control batching.