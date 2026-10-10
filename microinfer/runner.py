"""M08 exercise: execute packed work without mutating scheduler-owned sequences."""

from microinfer.engine.contracts import SchedulerOutput


class ModelRunner:
    def __init__(
        self, model, kv_cache, sampler, *, block_size: int, device="cpu", greedy=False
    ):
        self.model = model
        self.kv_cache = kv_cache
        self.sampler = sampler
        self.block_size = block_size
        self.device = device
        self.greedy = greedy

    def execute(self, output: SchedulerOutput) -> dict[int, int]:
        """prepare_batch -> model.forward_batch(batch, kv_cache) -> sample rows.

        Model returns logits [T,V] and writes per-layer KV. Run in inference mode
        with model.eval(). Only sample=True items produce Python int tokens.
        Empty work makes no model call. Do not advance counters or publish hashes.
        Propagate model errors so EngineCore does not commit failed execution.
        """
        raise NotImplementedError("M08: ModelRunner.execute")
