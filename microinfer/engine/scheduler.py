"""M01 exercise: admission, token budgets and request lifecycle.

Policy: running requests first (arrival order), then waiting requests (FCFS).
Running includes chunked prefills. Reserve prompt blocks via existing BlockManager.
Exactly one schedule may be in flight; commit only after successful execution.
"""

from microinfer.config.engine import EngineConfig
from microinfer.engine.blockmanager import BlockManager
from microinfer.engine.contracts import RequestOutput, SchedulerOutput
from microinfer.engine.sequence import Sequence


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager

    def add(self, seq: Sequence) -> None:
        """Validate admission, reject duplicate IDs, enqueue without allocating KV."""
        raise NotImplementedError("M01: Scheduler.add")

    def schedule(self) -> SchedulerOutput:
        """Reserve slots, set num_scheduled_tokens, return immutable work snapshot.

        Do not advance num_cached_tokens or publish hashes here. Under decode
        pressure preempt newest other running requests; never evict work already
        selected in this batch. If alone and physically unable to progress, raise
        RuntimeError instead of looping forever. Empty engine returns empty work.
        """
        raise NotImplementedError("M01: Scheduler.schedule")

    def update(
        self, output: SchedulerOutput, sampled_tokens: dict[int, int]
    ) -> list[RequestOutput]:
        """Validate the exact sample-ID set before mutations; reject stale work.

        hash_blocks -> advance cached count -> reset scheduled count -> append
        sample -> check EOS/limits -> release if finished. No sample/event for
        incomplete prefill chunks. Returned token_ids are cumulative completions.
        """
        raise NotImplementedError("M01: Scheduler.update")

    def preempt(self, seq_id: int) -> None:
        """Release KV, retain all token IDs, reset counters, queue for recompute."""
        raise NotImplementedError("M01: Scheduler.preempt")

    def abort(self, seq_id: int) -> bool:
        """Release/remove a request; unknown IDs return False. In-flight IDs error."""
        raise NotImplementedError("M01: Scheduler.abort")

    def has_unfinished_requests(self) -> bool:
        raise NotImplementedError("M01: Scheduler.has_unfinished_requests")
