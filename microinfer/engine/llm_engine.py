from microinfer.engine.sequence import SequenceStatus


class LLMEngine:
    def __init__(self, scheduler, runner):
        self.scheduler, self.model_runner = scheduler, runner

    def add_request(self, seq):
        self.scheduler.add(seq)

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        seqs, prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        num_tokens = (
            sum(s.num_scheduled_tokens for s in seqs) if prefill else -len(seqs)
        )
        try:
            tokens = self.model_runner.run(seqs, prefill)
        except Exception:
            # KV may have been partially written. Abort before publishing hashes.
            for seq in seqs:
                self.scheduler.abort(seq.seq_id)
            raise
        self.scheduler.postprocess(seqs, tokens, prefill)
        return [
            (s.seq_id, s.completion_token_ids)
            for s in seqs
            if s.status == SequenceStatus.FINISHED
        ], num_tokens

    def abort(self, seq_id):
        return self.scheduler.abort(seq_id)
