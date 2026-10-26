from microinfer.engine.sequence import SequenceStatus


class LLMEngine:
    def __init__(self, scheduler, runner):
        self.scheduler, self.model_runner = scheduler, runner

    def add_request(self, seq):
        self.scheduler.add(seq)

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        finished, num_tokens, _ = self.step_events()
        return finished, num_tokens

    def step_events(self):
        """Run one step and return per-request completion snapshots.

        ``step`` keeps the two-tuple API used by the benchmark. Streaming
        frontends use this method to observe tokens (and finish reasons) for
        every scheduled request, not just the ones that completed this step.
        """
        seqs, prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0, []
        num_tokens = (
            sum(s.num_scheduled_tokens for s in seqs) if prefill else -len(seqs)
        )
        try:
            if prefill and any(not seq.is_prefill for seq in seqs):
                # Mixed step: keep running decode on the graph/compiled path
                # and run the prefill chunk separately. Concatenating logits
                # would couple two different attention layouts and kernels.
                by_id = {}
                decode_seqs = [seq for seq in seqs if not seq.is_prefill]
                prefill_seqs = [seq for seq in seqs if seq.is_prefill]
                if decode_seqs:
                    decoded = self.model_runner.run(decode_seqs, False)
                    by_id.update(
                        (seq.seq_id, token) for seq, token in zip(decode_seqs, decoded)
                    )
                if prefill_seqs:
                    prefilled = self.model_runner.run(prefill_seqs, True)
                    by_id.update(
                        (seq.seq_id, token)
                        for seq, token in zip(prefill_seqs, prefilled)
                    )
                tokens = [by_id[seq.seq_id] for seq in seqs]
            else:
                tokens = self.model_runner.run(seqs, prefill)
        except Exception:
            # KV may have been partially written. Abort before publishing hashes.
            for seq in seqs:
                self.scheduler.abort(seq.seq_id)
            raise
        self.scheduler.postprocess(seqs, tokens, prefill)
        finished = [
            (s.seq_id, s.completion_token_ids)
            for s in seqs
            if s.status == SequenceStatus.FINISHED
        ]
        updates = [
            (
                s.seq_id,
                tuple(s.completion_token_ids),
                s.status == SequenceStatus.FINISHED,
                s.finish_reason,
            )
            for s in seqs
        ]
        return finished, num_tokens, updates

    def stats(self):
        stats = self.scheduler.stats()
        stats.update(self.scheduler.block_manager.stats())
        return stats

    def abort(self, seq_id):
        return self.scheduler.abort(seq_id)
