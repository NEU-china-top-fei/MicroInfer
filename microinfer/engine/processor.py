"""Tokenizer boundary and cumulative completion decoding."""

from microinfer.config import SampleParameter
from microinfer.engine.contracts import RequestOutput, TextOutput
from microinfer.engine.sequence import Sequence


class InputProcessor:
    def __init__(self, tokenizer, *, max_model_len: int, vocab_size: int | None = None):
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size

    def process(
        self, prompt: str | list[int], sampling_params: SampleParameter
    ) -> Sequence:
        """Encode strings with add_special_tokens=True; accept copied token lists.

        Reject empty/negative/non-integer tokens, prompt >= max_model_len and
        max_tokens <= 0 with ValueError. No implicit truncation or chat template.
        """
        tokens = (
            self.tokenizer.encode(prompt, add_special_tokens=True)
            if isinstance(prompt, str)
            else prompt
        )
        if (
            not isinstance(tokens, (list, tuple))
            or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or len(tokens) >= self.max_model_len
            or sampling_params.max_tokens <= 0
        ):
            raise ValueError("invalid prompt tokens or generation length")
        vocab_size = self.vocab_size
        if vocab_size is not None and any(t >= vocab_size for t in tokens):
            raise ValueError("token ID exceeds tokenizer vocabulary")
        return Sequence(list(tokens), sampling_params)


class OutputProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, output: RequestOutput) -> TextOutput:
        """Decode the entire completion with skip_special_tokens=True each time.

        Do not concatenate separately decoded tokens (UTF-8 pieces may span IDs).
        Keep EOS in output.token_ids even if the tokenizer omits it in text.
        """
        return TextOutput(
            output.seq_id,
            output.token_ids,
            self.tokenizer.decode(list(output.token_ids), skip_special_tokens=True),
            output.finished,
            output.finish_reason,
        )
