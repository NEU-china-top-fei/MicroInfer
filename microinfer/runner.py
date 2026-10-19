"""M02/M08: nano-vllm ModelRunner public methods with injected CPU test seams."""

from microinfer.engine.sequence import Sequence
from microinfer.utils.context import set_context
import torch


class ModelRunner:
    def __init__(self, model, kv_cache, sampler, *, block_size: int, device="cpu"):
        self.model = model
        self.kv_cache = kv_cache
        self.sampler = sampler
        self.block_size = block_size
        self.device = device

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Return flat input_ids/positions and set Context with query/key offsets,
        maximum lengths and physical slots. block_tables is None without cached
        prefixes; context_lens is None. Each seq uses its scheduled token range.
        """
        input_ids = []
        positions = []
        cumulative_q_offset = [0]
        cumulative_k_offset = [0]
        max_seq_len_q = 0
        max_seq_len_k = 0
        block_table = None
        slot_map = []
        for seq in seqs:
            this_seq_begin_idx = seq.num_cached_tokens
            this_seq_end_idx = seq.num_scheduled_tokens + this_seq_begin_idx
            input_ids.extend(seq[this_seq_begin_idx:this_seq_end_idx])
            positions.extend(range(this_seq_begin_idx, this_seq_end_idx))

            if not seq.block_table:
                continue  # warm up

            cur_q_len = seq.num_scheduled_tokens  # this time only compute input
            cur_k_len = this_seq_end_idx  # key must involve whole seq
            max_seq_len_q = max(max_seq_len_q, cur_q_len)
            max_seq_len_k = max(max_seq_len_k, cur_k_len)
            cumulative_q_offset.append(cumulative_q_offset[-1] + cur_q_len)
            cumulative_k_offset.append(cumulative_k_offset[-1] + cur_k_len)

            # slot_mapping:the ith  token 's cache are actually located where in all KV cache
            # block_num = (
            #     seq.num_scheduled_tokens + self.block_size - 1
            # ) // self.block_size
            # straight calculate the block_num will loss the actual num,can be the case that less that size tokens are distributed in two blocks
            block_offset = this_seq_begin_idx // self.block_size
            block_end = (this_seq_end_idx + self.block_size - 1) // self.block_size
            for i in range(block_offset, block_end):
                slot_begin = seq.block_table[i] * self.block_size
                if i == 0:
                    slot_begin += this_seq_begin_idx % self.block_size

                if i == block_end - 1:
                    slot_end = slot_begin + this_seq_end_idx % self.block_size
                else:
                    slot_end = (slot_begin // self.block_size + 1) * self.block_size

                slot_map.extend(range(slot_begin, slot_end))

            if cumulative_k_offset[-1] > cumulative_q_offset[-1]:
                """
                prefix cache:the only way that the above condition represent false is k_offset==q,which means no prefix cache
                and no chunked cache
                e.g: seqA[5~12]->q=12-5=7 while k=12,where prefix cache already happen
                """
                block_table = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cumulative_q_offset = torch.tensor(
            cumulative_q_offset, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cumulative_k_offset = torch.tensor(
            cumulative_k_offset, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_map = torch.tensor(slot_map, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        # pin_memory: can't be page out on CPU memory
        # non_block:asynchronize
        set_context(
            True,
            cu_seqlens_q=cumulative_q_offset,
            cu_seqlens_k=cumulative_k_offset,
            max_seqlen_q=max_seq_len_q,
            max_seqlen_k=max_seq_len_k,
            slot_mapping=slot_map,
            context_lens=None,
            block_tables=block_table,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Return last tokens / len(seq)-1, set decode Context with context_lens,
        slot_mapping and padded block_tables. Query/key cumulative arrays absent.
        """
        inputs = []
        positions = []
        slot_mapping = []
        context_len = []  # current whole sequence length of every seq
        block_table = self.prepare_block_tables(seqs)
        for seq in seqs:
            inputs.append(seq.last_token)
            positions.append(seq.num_tokens - 1)
            context_len.append(seq.num_tokens)
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )  # the last token that lies in this position
        inputs = torch.tensor(inputs, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_len = torch.tensor(
            context_len, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(False, None, None, 0, 0, slot_mapping, context_len, block_table)

        return inputs, positions

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
