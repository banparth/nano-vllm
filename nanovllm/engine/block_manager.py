from collections import deque
from typing import Any
import xxhash
import numpy as np
from enum import Enum
from collections import OrderedDict

from nanovllm.engine.sequence import Sequence
from nanovllm.config import Config

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class CPUBlock:
    # class Place(Enum):
    #     NONE = 0
    #     GPU = 1
    #     CPU = 2
    
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.token_ids = []

def _compute_hash(token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()

class SimpleCPUCacheBlockManager:
    def __init__(self, config: Config):
        self.block_size = config.kvcache_block_size
        self.blocks_gpu: list[Block] = [Block(i) for i in range(config.num_kvcache_blocks)]
        self.blocks_cpu: list[CPUBlock] = [CPUBlock(i) for i in range(config.num_cpu_kvcache_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.hash_to_block_cpu_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(config.num_kvcache_blocks))
        self.used_block_ids: set[int] = set()
        self.free_cpu_block_ids: deque[int] = deque(range(config.num_cpu_kvcache_blocks))
        self.used_cpu_block_ids: OrderedDict[int, None] = OrderedDict()
        
        self.move_cpu_to_gpu: list[tuple[int, int]] = []
        self.move_gpu_to_cpu: list[tuple[int, int]] = []

        self._reset_stats()

    def _reset_stats(self):
        # Per-pass instrumentation. Reset by pop_stats().
        # All counters are in BLOCKS (multiply by block_size for tokens).
        self.stats: dict[str, int] = {
            "gpu_hits": 0,         # prefix blocks reused from GPU (no copy)
            "cpu_hits": 0,         # prefix blocks recovered from CPU (queued H2D)
            "misses": 0,           # prefix/suffix blocks freshly allocated (uncached)
            "evictions": 0,        # GPU blocks pushed out to CPU (queued D2H)
            "cpu_lru_drops": 0,    # CPU entries dropped by LRU to make room
            "prefill_seqs": 0,     # sequences seen by allocate() this pass
            "h2d_copies": 0,       # CPU->GPU block copies actually queued
            "d2h_copies": 0,       # GPU->CPU block copies actually queued
        }

    def pop_stats(self) -> dict[str, int]:
        s = self.stats.copy()
        self._reset_stats()
        return s

    def compute_hash(self, token_ids: list[int], prefix: int = -1):
        return _compute_hash(token_ids, prefix)
    
    def can_allocate(self, seq: Sequence) -> int:
        '''
        Check if we can allocate the blocks for the sequence.
        Return the number of cached blocks if we can allocate, otherwise return -1.
        '''
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks_gpu[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def _deallocate_block(self, block_id: int):
        assert self.blocks_gpu[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks_gpu[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
            if len(self.free_cpu_block_ids) > 0:
                cpu_block_id = self.free_cpu_block_ids.popleft()
                self.used_cpu_block_ids[cpu_block_id] = None
                self.hash_to_block_cpu_id[block.hash] = cpu_block_id
                self.blocks_cpu[cpu_block_id].update(block.hash, block.token_ids)
                self.move_gpu_to_cpu.append((block_id, cpu_block_id))
                self.stats["d2h_copies"] += 1
            elif len(self.used_cpu_block_ids) > 0:
                cpu_block_id = self.used_cpu_block_ids.popitem(last=False)[0]
                self.used_cpu_block_ids[cpu_block_id] = None
                self.hash_to_block_cpu_id[block.hash] = cpu_block_id
                self.blocks_cpu[cpu_block_id].update(block.hash, block.token_ids)
                self.move_gpu_to_cpu.append((block_id, cpu_block_id))
                self.stats["cpu_lru_drops"] += 1
                self.stats["d2h_copies"] += 1
            self.stats["evictions"] += 1

        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        '''
        Allocate the blocks for the sequence.
        '''
        assert not seq.block_table
        self.stats["prefill_seqs"] += 1
        self.stats["gpu_hits"] += num_cached_blocks
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks_gpu[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        num_cpu_cached = 0
        for i in range(num_cached_blocks, seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            cpu_block_id = self.hash_to_block_cpu_id.get(h, -1)
            if cpu_block_id == -1 or self.blocks_cpu[cpu_block_id].token_ids != token_ids:
                break
            gpu_block_id = self._allocate_block()
            seq.block_table.append(gpu_block_id)
            self.move_cpu_to_gpu.append((cpu_block_id, gpu_block_id))
            self.stats["h2d_copies"] += 1
            self.blocks_gpu[gpu_block_id].update(h, token_ids)
            self.free_cpu_block_ids.append(cpu_block_id)
            self.used_cpu_block_ids.pop(cpu_block_id)
            self.hash_to_block_id[h] = gpu_block_id
            num_cpu_cached += 1
        self.stats["cpu_hits"] += num_cpu_cached

        num_miss = seq.num_blocks - num_cached_blocks - num_cpu_cached
        for i in range(num_cached_blocks + num_cpu_cached, seq.num_blocks):
            h = self.compute_hash(seq.block(i), h)
            seq.block_table.append(self._allocate_block())
        self.stats["misses"] += num_miss

        seq.num_cached_tokens = (num_cached_blocks + num_cpu_cached) * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks_gpu[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks_gpu[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks_gpu[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        return _compute_hash(token_ids, prefix)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
