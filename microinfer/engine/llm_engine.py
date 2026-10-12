"""M09: nano-vllm synchronous schedule/run/postprocess iteration.

Dependency injection avoids requiring a GPU/model directory for unit tests.
"""


class LLMEngine:
    def __init__(self, scheduler, runner):
        self.scheduler = scheduler
        self.model_runner = runner

    def add_request(self, seq):
        """Internal tokenized admission seam; the frontend handles text inputs."""
        raise NotImplementedError("M09: LLMEngine.add_request")

    def is_finished(self):
        raise NotImplementedError("M09: LLMEngine.is_finished")

    def step(self):
        """seqs, mode = schedule(); tokens = runner.run(seqs, mode);
        scheduler.postprocess(seqs, tokens, mode).
        Return ([(finished_seq_id, completion_token_ids), ...], num_tokens).
        Compute num_tokens BEFORE postprocess clears scheduled counts:
        positive sum(num_scheduled_tokens) for prefill; -len(seqs) for decode.
        No partial completion events. Do not postprocess failed execution.
        """
        raise NotImplementedError("M09: LLMEngine.step")

    def abort(self, seq_id):
        """Optional extension, not nano-vllm's core API."""
        raise NotImplementedError("extension: LLMEngine.abort")
