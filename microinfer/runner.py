"""M02/M08: nano-vllm ModelRunner public methods with injected CPU test seams."""

from microinfer.engine.sequence import Sequence


class ModelRunner:
    def __init__(self, model, kv_cache, sampler, *, block_size: int, device="cpu"):
        self.model = model
        self.kv_cache = kv_cache
        self.sampler = sampler
        self.block_size = block_size
        self.device = device

    def prepare_prefill(self, seqs: list[Sequence]):
        """Return flat input_ids/positions and set Context with query/key offsets,
        maximum lengths and physical slots. block_tables is None without cached
        prefixes; context_lens is None. Each seq uses its scheduled token range.
        """
        raise NotImplementedError("M02: ModelRunner.prepare_prefill")

    def prepare_decode(self, seqs: list[Sequence]):
        """Return last tokens / len(seq)-1, set decode Context with context_lens,
        slot_mapping and padded block_tables. Query/key cumulative arrays absent.
        """
        raise NotImplementedError("M02: ModelRunner.prepare_decode")

    def prepare_sample(self, seqs: list[Sequence]):
        """Return per-sequence float32 temperatures on self.device."""
        raise NotImplementedError("M08: ModelRunner.prepare_sample")

    def run_model(self, input_ids, positions, is_prefill: bool):
        """Eager MVP: model(input_ids, positions) -> hidden -> compute_logits.
        Return [len(seqs), vocab] logits. Prefill LM head selects the last query
        per sequence using context.cu_seqlens_q[1:]-1; decode uses every row.
        The selection belongs to the model/LM head, as in nano-vllm.
        """
        raise NotImplementedError("M08: ModelRunner.run_model")

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """prepare_prefill/decode -> prepare_sample -> run_model -> sampler(...).

        Return one Python token ID per sequence, including incomplete chunks.
        Scheduler.postprocess decides whether to discard a chunk's sample.
        Run in inference mode; model.eval(); reset Context after execution.
        Empty input and exception cleanup are MicroInfer extension tests.
        """
        raise NotImplementedError("M08: ModelRunner.run")
