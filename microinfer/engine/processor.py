"""M10 exercise: tokenizer boundary and cumulative completion decoding."""

from microinfer.config import SampleParameter
from microinfer.engine.contracts import RequestOutput, TextOutput
from microinfer.engine.sequence import Sequence


class InputProcessor:
    def __init__(self, tokenizer, *, max_model_len: int):
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len

    def process(
        self, prompt: str | list[int], sampling_params: SampleParameter
    ) -> Sequence:
        """Encode strings with add_special_tokens=True; accept copied token lists.

        Reject empty/negative/non-integer tokens, prompt >= max_model_len and
        max_tokens <= 0 with ValueError. No implicit truncation or chat template.
        """
        raise NotImplementedError("M10: InputProcessor.process")


class OutputProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, output: RequestOutput) -> TextOutput:
        """Decode the entire completion with skip_special_tokens=True each time.

        Do not concatenate separately decoded tokens (UTF-8 pieces may span IDs).
        Keep EOS in output.token_ids even if the tokenizer omits it in text.
        """
        raise NotImplementedError("M10: OutputProcessor.process")
