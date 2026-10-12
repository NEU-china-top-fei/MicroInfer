"""M10: nano-vllm generate output contract; injected frontend helpers for tests."""


class LLM:
    def __init__(self, engine, input_processor, output_processor):
        self.engine = engine
        self.input_processor = input_processor
        self.output_processor = output_processor

    def generate(self, prompts, sampling_params, use_tqdm=True):
        """Accept one SampleParameter or a per-prompt list. Step until is_finished.
        Return [{"text": str, "token_ids": list[int]}, ...] in prompt order.
        Consume only finished tuples from engine.step(); never stream partials.
        Empty input returns []. All-input validation is an optional extension.
        """
        raise NotImplementedError("M10: LLM.generate")
