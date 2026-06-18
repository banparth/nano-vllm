#pyright: reportMissingTypeStubs=false
from typing import Any
import torch
import numpy as np
from typing import final
from cuda.bindings import driver as cu
import psutil
from nanovllm.config import Config
import torch.nn as nn

@final
class SimpleCPUCacheRunner:
    num_hidden_layers: int
    num_cpu_kvcache_blocks: int
    num_kvcache_blocks: int
    block_size: int
    def __init__(self, config: Config, model: nn.Module):
        # get config
        self.config = config
        hf_config = config.hf_config
        
        # get block size in bytes
        self.block_size = config.kvcache_block_size
        num_kv_heads = hf_config.num_key_value_heads // config.tensor_parallel_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # get gpu num kv cache blocks
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        self.num_kvcache_blocks = config.num_kvcache_blocks
        assert config.num_kvcache_blocks > 0    
    
        # get cpu num kv cache blocks
        vm = psutil.virtual_memory()
        cpu_total = vm.total
        # cpu_available = vm.available
        # our_rss = psutil.Process().memory_info().rss
        # cpu_used_by_others = cpu_total - cpu_available - our_rss
        
        
        cpu_budget = int(cpu_total*self.config.cpu_memory_utilization)
        cpu_budget = max(cpu_budget, 0) // config.tensor_parallel_size
        config.num_cpu_kvcache_blocks = cpu_budget // block_bytes
        self.num_cpu_kvcache_blocks = config.num_cpu_kvcache_blocks
        # allocate gpu and cpu cache
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        
        self.gpu_cache = torch.empty(2, self.num_hidden_layers, self.num_kvcache_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        self.cpu_cache = torch.empty(2, self.num_hidden_layers, self.num_cpu_kvcache_blocks, self.block_size, self.num_kv_heads, self.head_dim, device="cpu", pin_memory=True)

        # Dedicated copy stream. cuMemcpyBatchAsync rejects the legacy NULL
        # stream (which is what torch.cuda.current_stream() returns by default),
        # and using a side stream also lets H<->D copies overlap with compute.
        self.copy_stream = torch.cuda.Stream()

        # Precompute per-(kv, layer) base pointers and per-block stride in
        # bytes so move() can build copy descriptors with pure integer
        # arithmetic instead of repeatedly indexing the cache tensors.
        # Address of slice [kv, layer, block] = base[kv*L + layer] + block * block_stride.
        self._gpu_layer_base = [
            int(self.gpu_cache[kv, layer_id].data_ptr())
            for kv in range(2)
            for layer_id in range(self.num_hidden_layers)
        ]
        self._cpu_layer_base = [
            int(self.cpu_cache[kv, layer_id].data_ptr())
            for kv in range(2)
            for layer_id in range(self.num_hidden_layers)
        ] if self.num_cpu_kvcache_blocks > 0 else []
        self._block_stride_bytes = (
            self.block_size * self.num_kv_heads * self.head_dim * self.gpu_cache.element_size()
        )

        # assign gpu cache to model
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.gpu_cache[0, layer_id]
                module.v_cache = self.gpu_cache[1, layer_id]
                layer_id += 1


    def move(self, move_cpu_to_gpu: list[tuple[int, int]], move_gpu_to_cpu: list[tuple[int, int]]) -> torch.cuda.Event:
        n = 2 * len(move_cpu_to_gpu) * self.num_hidden_layers + 2 * len(move_gpu_to_cpu) * self.num_hidden_layers
        block_stride = self._block_stride_bytes
        gpu_base = self._gpu_layer_base
        cpu_base = self._cpu_layer_base
        L = self.num_hidden_layers
        CUdeviceptr = cu.CUdeviceptr

        # Build dsts/srcs via integer arithmetic. About 20x cheaper than
        # tensor-view + data_ptr() per element on H100/Qwen3-0.6B.
        dsts: list[Any] = [
            CUdeviceptr(gpu_base[kv * L + layer_id] + move_cpu_to_gpu[i][1] * block_stride)
            for layer_id in range(L)
            for kv in range(2)
            for i in range(len(move_cpu_to_gpu))
        ] + [
            CUdeviceptr(cpu_base[kv * L + layer_id] + move_gpu_to_cpu[i][1] * block_stride)
            for layer_id in range(L)
            for kv in range(2)
            for i in range(len(move_gpu_to_cpu))
        ]
        srcs = [
            CUdeviceptr(cpu_base[kv * L + layer_id] + move_cpu_to_gpu[i][0] * block_stride)
            for layer_id in range(L)
            for kv in range(2)
            for i in range(len(move_cpu_to_gpu))
        ] + [
            CUdeviceptr(gpu_base[kv * L + layer_id] + move_gpu_to_cpu[i][0] * block_stride)
            for layer_id in range(L)
            for kv in range(2)
            for i in range(len(move_gpu_to_cpu))
        ]
        sizes = [block_stride] * n

        # Same srcAccessOrder/flags work for both H2D and D2H copies, so one
        # CUmemcpyAttributes covers the whole batch. Split into two only if
        # you need direction-specific srcAccessOrder or flags.
        attr = cu.CUmemcpyAttributes()
        attr.srcAccessOrder = (
            cu.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        )
        attr.flags = 0

        # Make the copy stream wait for in-flight attention writes (which run
        # on the compute stream) before evicting/loading those GPU blocks.
        compute_stream = torch.cuda.current_stream()
        self.copy_stream.wait_stream(compute_stream)

        (err,) = cu.cuMemcpyBatchAsync(
            dsts,
            srcs,
            sizes,
            n,
            [attr],
            [0],
            1,
            self.copy_stream.cuda_stream,
        )
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyBatchAsync failed with error code {err}")
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        return event
