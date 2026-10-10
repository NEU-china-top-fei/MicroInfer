"""M09 exercise: synchronous single-process schedule/execute/commit loop."""


class LLMEngine:
    def __init__(self, scheduler, runner):
        self.scheduler = scheduler
        self.runner = runner

    def add_request(self, seq):
        raise NotImplementedError("M09: LLMEngine.add_request")

    def step(self):
        """Empty work returns []; otherwise execute before update.

        Propagate execution errors and never commit/hash incomplete KV. MVP errors
        are fatal for the engine instance; automatic retry is outside this scope.
        """
        raise NotImplementedError("M09: LLMEngine.step")

    def abort(self, seq_id):
        raise NotImplementedError("M09: LLMEngine.abort")

    def has_unfinished_requests(self):
        raise NotImplementedError("M09: LLMEngine.has_unfinished_requests")
