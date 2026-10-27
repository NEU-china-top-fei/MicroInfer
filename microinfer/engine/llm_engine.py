from microinfer.engine.sequence import SequenceStatus
from microinfer.engine.speculative import NgramSpeculator


class LLMEngine:
    def __init__(self, scheduler, runner, speculator=None):
        self.scheduler, self.model_runner = scheduler, runner
        config = getattr(scheduler, "config", None)
        if speculator is not None:
            self.speculator = speculator
        elif config is not None and config.use_speculative:
            self.speculator = NgramSpeculator(
                ngram=config.speculative_ngram,
                max_drafts=config.speculative_max_drafts,
            )
        else:
            self.speculator = None

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
        every scheduled request, including the ones still running after this
        step.
        """
        if self.speculator is not None:
            speculative = self._speculative_step()
            if speculative is not None:
                return speculative
        seqs, prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0, []
        return self._execute_scheduled(seqs, prefill)

    def _execute_scheduled(self, seqs, prefill):
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
        for seq_id, _ in finished:
            self._release_spec(seq_id)
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

    def _rollback_spec(self, proposals):
        for seq, drafts, base in proposals:
            seq.token_ids = seq.token_ids[:base]
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.spec_draft_ids = ()
            seq.spec_base_len = 0
            seq.num_scheduled_tokens = 0
            seq.is_prefill = False
            self.scheduler.block_manager.trim(seq)

    def _speculative_step(self):
        """Try one greedy n-gram verification step.

        Speculation is opportunistic and only runs when there is no waiting
        prefill. Any missing draft or insufficient KV falls back to the normal
        scheduler path for the whole step.
        """
        scheduler = self.scheduler
        config = scheduler.config
        if (
            scheduler._inflight is not None
            or scheduler.waiting
            or not scheduler.running
        ):
            return None
        running = list(scheduler.running)
        if any(not seq.greedy for seq in running):
            return None

        limits = []
        for seq in running:
            base = len(seq)
            remaining = min(
                config.speculative_max_drafts,
                seq.max_tokens - seq.num_completion_tokens - 1,
                scheduler.max_length - base - 1,
            )
            if remaining <= 0:
                return None
            limits.append(remaining)
        propose_batch = getattr(self.speculator, "propose_batch", None)
        if propose_batch is not None:
            draft_lists = propose_batch(running, limits)
        else:
            draft_lists = [
                self.speculator.propose(seq, limit)
                for seq, limit in zip(running, limits)
            ]
        proposals = []
        for seq, limit, drafts in zip(running, limits, draft_lists):
            drafts = list(drafts)[:limit]
            if not drafts:
                return None
            proposals.append((seq, drafts, len(seq)))

        for seq, drafts, _ in proposals:
            seq.token_ids.extend(drafts)
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
        needed = sum(
            max(0, seq.num_blocks - len(seq.block_table)) for seq, _, _ in proposals
        )
        if needed > len(scheduler.block_manager.free_block_ids):
            self._rollback_spec(proposals)
            return None
        for seq, drafts, base in proposals:
            scheduler.block_manager.may_append(seq)
            seq.spec_base_len = base
            seq.spec_draft_ids = tuple(drafts)
            seq.num_scheduled_tokens = 1 + len(drafts)
            seq.is_prefill = True

        seqs = [seq for seq, _, _ in proposals]
        scheduler.record_spec_attempt(seqs)
        try:
            metadata, logits = self.model_runner.run_verify(seqs)
            accepted = []
            cu_q = metadata.cu_q_cpu
            for index, seq in enumerate(seqs):
                rows = logits[cu_q[index] : cu_q[index + 1]]
                accepted.append(self._accept_greedy(seq, rows))
        except Exception:
            for seq in seqs:
                scheduler.abort(seq.seq_id)
            raise
        scheduler.commit_spec(seqs, accepted)

        # Generate the next token after the accepted prefix with the normal
        # decode kernel. Verification logits only select the accepted draft
        # prefix; the emitted continuation token still comes from decode.
        if scheduler.running:
            normal_seqs, normal_prefill = scheduler.schedule()
            if normal_seqs:
                self._execute_scheduled(normal_seqs, normal_prefill)

        finished = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.status == SequenceStatus.FINISHED
        ]
        for seq_id, _ in finished:
            self._release_spec(seq_id)
        updates = [
            (
                seq.seq_id,
                tuple(seq.completion_token_ids),
                seq.status == SequenceStatus.FINISHED,
                seq.finish_reason,
            )
            for seq in seqs
        ]
        return finished, -len(seqs), updates

    def _accept_greedy(self, seq, rows):
        drafts = seq.spec_draft_ids
        margin = self.scheduler.config.speculative_min_margin
        accepted = 0
        for index, draft in enumerate(drafts):
            row = rows[index]
            top = row.topk(2)
            gap = float((top.values[0] - top.values[1]).item())
            target = int(top.indices[0])
            if gap < margin or target != draft:
                return accepted
            if not seq.ignore_eos and target == self.scheduler.eos:
                return accepted
            accepted += 1
        return accepted

    def stats(self):
        stats = self.scheduler.stats()
        stats.update(self.scheduler.block_manager.stats())
        return stats

    def _release_spec(self, seq_id):
        release = getattr(self.speculator, "release", None)
        if release is not None:
            release(seq_id)

    def abort(self, seq_id):
        self._release_spec(seq_id)
        return self.scheduler.abort(seq_id)
