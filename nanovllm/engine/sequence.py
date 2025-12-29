from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto() # queued, not yet scheduled
    RUNNING = auto() # scheduled, running
    FINISHED = auto() # finished, no more tokens to generate (eos or max_tokens)


class Sequence:
    block_size = 256 # kv cache block size
    counter = count() # class level counter for sequence ids

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter) # can I use self.counter? #NOTE
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids) # defensive copy
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return len(self.num_cached_tokens) // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens % self.block_size
        # return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size] # NOTE: this is a view, not a copy

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
            self.token_ids, self.num_completion_tokens == 0 else self.last_token)
        
    def __setstate__(self, state):
        ''' Optimized serialization for multiprocessing:
__getstate__: Saves only essential data. If no completion tokens, saves full token_ids; otherwise saves only last_token.
__setstate__: Restores state. If no completion tokens, restores full token_ids; otherwise restores last_token.
Why this optimization: For long sequences, storing only the last token reduces memory when sending sequences between processes.'''
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.token_ids = state[-1]

# - Scheduler uses Sequence objects to track requests.
# - BlockManager uses block_table to map logical to physical blocks.
# - ModelRunner uses token_ids and block methods for inference.
# - LLMEngine uses is_finished to check completion.
# This is the core data structure representing a single generation request throughout its lifecycle.
