"""M10 exercise: offline generation entrypoint, with dependencies injected."""


class LLM:
    def __init__(self, engine, input_processor, output_processor):
        self.engine = engine
        self.input_processor = input_processor
        self.output_processor = output_processor

    def generate(self, prompts, sampling_params):
        """Return final TextOutputs in input order, regardless of finish order.

        Validate/process all prompts before enqueueing any. Drain engine.step(),
        retain only finished events, decode completions. Empty prompts returns [].
        No model download, HTTP server or tokenizer construction in this method.
        """
        raise NotImplementedError("M10: LLM.generate")
