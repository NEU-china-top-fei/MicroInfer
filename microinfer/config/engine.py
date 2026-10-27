"""Single-process engine limits, independent of model dimensions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 512
    # Upper bound for one prefill chunk; keeps a long prompt from monopolizing
    # a step and stalling running decode requests.
    max_num_prefill_tokens: int = 512
    max_model_len: int = 4096
    eos_token_id: int | None = None
    # P1 safety net: keep expensive invariant checks on for tests/debugging.
    debug_mode: bool = False
    # P1.5 experiment switch; decode-only torch.compile.
    use_compile: bool = False
    # P2: capture decode buckets with CUDA Graph (eager capture; no compile).
    use_cuda_graph: bool = False
    # Compile RMSNorm/SwiGLU/RoPE individually; compatible with CUDA Graph.
    compile_ops: bool = False
    # P4: fill the token budget with waiting prefill after running decode.
    # Disable for the pre-P4 prefill-first A/B baseline.
    mixed_batch: bool = True
    # P5: n-gram prompt-lookup speculative decoding for greedy requests.
    use_speculative: bool = False
    speculative_max_drafts: int = 8
    speculative_ngram: int = 3
    # Reject drafts whose target top-1/top-2 gap is too small; prefill and
    # decode kernels can disagree at near-ties.
    speculative_min_margin: float = 0.5

    def __post_init__(self):
        for name in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_num_prefill_tokens",
            "max_model_len",
            "speculative_max_drafts",
            "speculative_ngram",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.speculative_min_margin) not in (int, float)
            or self.speculative_min_margin < 0
        ):
            raise ValueError("speculative_min_margin must be nonnegative")
        if self.eos_token_id is not None and (
            type(self.eos_token_id) is not int or self.eos_token_id < 0
        ):
            raise ValueError("eos_token_id must be a nonnegative integer or None")
        for name in (
            "debug_mode",
            "use_compile",
            "use_cuda_graph",
            "compile_ops",
            "mixed_batch",
            "use_speculative",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.use_compile and self.use_cuda_graph:
            raise ValueError("use_compile and use_cuda_graph are mutually exclusive")
        if self.use_compile and self.compile_ops:
            raise ValueError("use_compile and compile_ops are mutually exclusive")
