import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from nanovllm.engine.cache_connector.base import KVConnectorBase, KVConnectorRole
from nanovllm.engine.request import Request
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.use_breakable = config.use_breakable_cudagraph and not self.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.connector = KVConnectorBase(config, KVConnectorRole.WORKER)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        requests = [Request([0] * seq_len) for _ in range(num_seqs)]
        for request in requests:
            request.num_scheduled_tokens = seq_len
        self.run(requests, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        kv_caches: dict[str, torch.Tensor] = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                kv_caches[name] = self.kv_cache[:, layer_id]
                layer_id += 1
        self.connector.register_kv_caches(kv_caches)

    def prepare_block_tables(self, requests: list[Request]):
        max_len = max(len(request.block_table) for request in requests)
        block_tables = [
            request.block_table + [-1] * (max_len - len(request.block_table))
            for request in requests
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        return block_tables

    def prepare_prefill(self, requests: list[Request]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for request in requests:
            start = request.num_computed_tokens
            seqlen_q = request.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(request[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not request.block_table:  # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = request.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = request.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = request.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(requests)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, requests: list[Request]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for request in requests:
            input_ids.append(request.last_token)
            positions.append(len(request) - 1)
            context_lens.append(len(request))
            slot_mapping.append(
                request.block_table[-1] * self.block_size + request.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        block_tables = self.prepare_block_tables(requests)
        set_context(
            False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables
        )
        return input_ids, positions

    def prepare_sample(self, requests: list[Request]):
        temperatures = [request.temperature for request in requests]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(
            non_blocking=True
        )
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            pad = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[pad]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
            if self.use_breakable:
                # Decode attention runs eagerly during replay and reads the live
                # context, so point it at the padded static buffers.
                set_context(
                    False,
                    slot_mapping=graph_vars["slot_mapping"][:pad],
                    context_lens=graph_vars["context_lens"][:pad],
                    block_tables=graph_vars["block_tables"][:pad],
                )
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, requests: list[Request], is_prefill: bool, connector_meta=None) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(requests) if is_prefill else self.prepare_decode(requests)
        )
        if connector_meta is not None:
            self.connector.bind_connector_metadata(connector_meta)
            self.connector.start_load_kv(get_context())
        temperatures = self.prepare_sample(requests) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        if connector_meta is not None:
            self.connector.wait_for_save()
            self.connector.clear_connector_metadata()
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        if self.use_breakable:
            # One shared pool + capture stream across all segments/batch sizes.
            # (o_buf is allocated once in allocate_kv_cache, shared by all layers.)
            self.graph_pool = torch.cuda.graph_pool_handle()
            self.capture_stream = torch.cuda.Stream()

        for bs in reversed(self.graph_bs):
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            if self.use_breakable:
                torch.cuda.synchronize()
                bcg = BreakableCUDAGraph()
                with BreakableCUDAGraphCapture(
                    bcg, pool=self.graph_pool, stream=self.capture_stream
                ):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
                self.graphs[bs] = bcg
            else:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
