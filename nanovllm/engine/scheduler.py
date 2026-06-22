from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.cache_connector.base import KVConnectorBase, KVConnectorRole
from nanovllm.engine.request import Request, RequestStatus


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.connector = KVConnectorBase(config, KVConnectorRole.SCHEDULER)
        self.connector.bind_gpu_block_pool(self.block_manager)
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, request: Request):
        self.waiting.append(request)

    def build_connector_meta(self, requests: list[Request], is_prefill: bool):
        return self.connector.build_connector_meta(requests, is_prefill)

    def schedule(self) -> tuple[list[Request], bool]:
        scheduled_requests = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_requests) < self.max_num_seqs:
            request = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not request.block_table:
                num_cached_blocks = self.block_manager.can_allocate(request)
                if num_cached_blocks == -1:
                    break
                num_tokens = request.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = request.num_tokens - request.num_computed_tokens
            if (
                remaining < num_tokens and scheduled_requests
            ):  # only allow chunked prefill for the first request
                break
            if not request.block_table:
                self.block_manager.allocate(request, num_cached_blocks)
            request.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += request.num_scheduled_tokens
            if request.num_computed_tokens + request.num_scheduled_tokens == request.num_tokens:
                request.status = RequestStatus.RUNNING
                self.waiting.popleft()
                self.running.append(request)
            scheduled_requests.append(request)

        if scheduled_requests:
            return scheduled_requests, True

        # decode
        while self.running and len(scheduled_requests) < self.max_num_seqs:
            request = self.running.popleft()
            while not self.block_manager.can_append(request):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(request)
                    break
            else:
                request.num_scheduled_tokens = 1
                request.is_prefill = False
                self.block_manager.may_append(request)
                scheduled_requests.append(request)
        assert scheduled_requests
        self.running.extendleft(reversed(scheduled_requests))
        return scheduled_requests, False

    def preempt(self, request: Request):
        request.status = RequestStatus.WAITING
        request.is_prefill = True
        self.block_manager.deallocate(request)
        self.waiting.appendleft(request)

    def postprocess(self, requests: list[Request], token_ids: list[int], is_prefill: bool):
        for request, token_id in zip(requests, token_ids):
            self.block_manager.hash_blocks(request)
            request.num_computed_tokens += request.num_scheduled_tokens
            request.num_scheduled_tokens = 0
            if is_prefill and request.num_computed_tokens < request.num_tokens:
                continue
            request.append_token(token_id)
            if (
                not request.ignore_eos and token_id == self.eos
            ) or request.num_output_tokens == request.max_tokens:
                request.status = RequestStatus.FINISHED
                self.block_manager.deallocate(request)
                self.running.remove(request)
