"""Incremental completion decoding with a stable-text buffer.

Tokenizers decode token IDs, not bytes. A UTF-8 character may span several
tokens, so decoding only the newest token can produce a replacement character
that is corrected by the next token. We decode the cumulative completion and
hold back trailing replacement characters until the sequence is finished.
"""

from __future__ import annotations


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_ids: tuple[int, ...] = ()
        self.text = ""

    def update(self, token_ids, *, finished: bool = False) -> str:
        """Return the newly stable text since the previous call."""
        token_ids = tuple(token_ids)
        self.token_ids = token_ids
        decoded = self.tokenizer.decode(list(token_ids), skip_special_tokens=True)
        if not finished:
            while decoded.endswith("�"):
                decoded = decoded[:-1]
        if decoded.startswith(self.text):
            delta = decoded[len(self.text) :]
        else:
            # A tokenizer correction changed an already emitted prefix. There
            # is no way to retract text in a stream; emit the corrected tail.
            delta = decoded
        self.text = decoded
        return delta
