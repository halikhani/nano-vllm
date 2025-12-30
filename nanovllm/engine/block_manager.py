''' How it works:
- BlockManager doesn't track sequences directly; it manages a pool of physical blocks.
- Each Sequence has a block_table that maps logical blocks to physical block IDs.
- Reference counting (ref_count) tracks how many sequences share a block.
- Prefix caching: multiple sequences can share the same physical block if content matches.
- Access pattern: block_manager.blocks[seq.block_table[i]] gets the physical block for logical block i.
- The block_table is the link between sequences and their KV cache blocks. '''

from collections import deque
import xxhash # fast hash function for keys
import numpy as np

from nanovllm.engine.seuence import Sequence

# individual KVC block 
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1 # for prefix caching
        self.token_ids = [] # tokens in the block

    
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids
    
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

# memory manager for KVC blocks
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()


    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int=-1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, 'little')) #  includes the previous block's hash (chained hashing).
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int):
        ''' Allocates a free block:
        Verifies ref_count is 0.
        Resets the block (ref_count = 1).
        Moves from free to used. '''

        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        ''' Deallocates a block:
        Verifies ref_count is 0.
        Moves from used to free.'''
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)


    def can_allocate(self, seq: Sequence):
        return len(self.free_block_ids) >= seq.num_blocks

    
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block[i]
            # if the block is not full, we don't need to compute the hash
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # allocate a new block
                block_id = self.free_block_ids[0] # get the first free block
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size 
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)


    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()


    def can_append(self, seq:Sequence):
        # return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) # NOTE: is this correct??
        
        needs_new_block = (len(seq) % self.block_size == 0)
        if needs_new_block:
            return len(self.free_block_ids) >= 1
        else:
            # Can append to current block, no new block needed
            return True

    
    def may_append(self, seq:Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # The last block was full, and we just added the first token to a new block.
            # Allocate a new block for the next token.
            assert last_block.hash != -1, "Last block should have a hash"
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # The last block just became full.
            # Compute its hash and cache it for prefix caching.
            assert last_block.hash != -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash != -1, "Last block should have a hash"

        ''' Example Trace (block_size = 256)
        len(seq) = 255: 255 % 256 = 255 → Case 3 (no action)
        len(seq) = 256: 256 % 256 = 0 → Case 2 (compute hash, cache block)
        len(seq) = 257: 257 % 256 = 1 → Case 1 (allocate new block)
        len(seq) = 258: 258 % 256 = 2 → Case 3 (no action)
        len(seq) = 512: 512 % 256 = 0 → Case 2 (compute hash, cache block)
        '''