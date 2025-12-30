from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# The Scheduler implements continuous batching: 
# it mixes prefill (new requests) and decode (ongoing generation) in the same batch. 
# It manages two queues: waiting (new requests) and running (active generation).
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]: # returns a batch of sequences and whether it's a prefill batch.
        # Prefill logic:
        # Take sequences from waiting (FIFO)
        # Check constraints:
        # If constraints pass:
            # Allocate blocks
            # Count tokens (excluding cached)
            # Move to running
            # Add to batch
        # Return batch if any sequences scheduled
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += (len(seq) - seq.num_cached_tokens)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # Decode logic:
        # Take sequences from running (FIFO)
        # If can_append(seq) fails:
        #     Preempt other running sequences (LIFO) to free blocks
        #     If none left, preempt the current sequence
        # If can_append(seq) succeeds:
        #     Call may_append(seq) to manage blocks
        #     Add to batch
        # Put scheduled sequences back at the front of running (preserving order)
        # Return batch
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) # preempt from right since that the last appended req (FIFO)
                else:
                    self.preempt(seq)
                    break

            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
        assert scheduled_seqs, "No sequences scheduled"
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        ''' Post-processing:
            - Append generated tokens
            - Check completion (EOS or max_tokens)
            - If finished: set FINISHED, deallocate blocks, remove from running '''

        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                


        

