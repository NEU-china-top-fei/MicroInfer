"""Async request frontend for the single-process engine.

The model runner is synchronous and owns CUDA state. A single background task
serializes ``step_events`` calls, while callers receive per-request queues for
streaming. Requests are admitted under the same lock as the step, so scheduler
mutation never races with model execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from microinfer.config import SampleParameter
from microinfer.engine.contracts import RequestOutput, StreamEvent
from microinfer.engine.detokenizer import IncrementalDetokenizer


class BackpressureError(RuntimeError):
    """Raised when the waiting/active request limit is reached."""


@dataclass
class AsyncRequest:
    seq_id: int
    prompt_tokens: int
    queue: asyncio.Queue = field(repr=False)
    created_at: float = field(default_factory=time.monotonic)
    token_ids: tuple[int, ...] = ()
    text: str = ""
    finished: bool = False
    finish_reason: str | None = None

    async def stream(self):
        while True:
            event = await self.queue.get()
            yield event
            if event.finished:
                break


class AsyncEngine:
    def __init__(self, llm, *, max_waiting: int = 128):
        if type(max_waiting) is not int or max_waiting <= 0:
            raise ValueError("max_waiting must be a positive integer")
        self.llm = llm
        self.engine = llm.engine
        self.input_processor = llm.input_processor
        self.output_tokenizer = llm.output_processor.tokenizer
        self.max_waiting = max_waiting
        self._lock = asyncio.Lock()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._requests: dict[int, AsyncRequest] = {}
        self._detokenizers: dict[int, IncrementalDetokenizer] = {}
        self._stopping = False
        self._error: BaseException | None = None

    def _ensure_task(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())

    async def add_request(self, prompt, sampling_params: SampleParameter):
        """Admit one request and return its streaming handle."""
        if not isinstance(sampling_params, SampleParameter):
            raise TypeError("sampling_params must be SampleParameter")
        async with self._lock:
            if self._stopping:
                raise RuntimeError("AsyncEngine is stopping")
            if len(self._requests) >= self.max_waiting:
                raise BackpressureError("too many active requests")
            seq = self.input_processor.process(prompt, sampling_params)
            self.engine.add_request(seq)
            request = AsyncRequest(
                seq_id=seq.seq_id,
                prompt_tokens=seq.num_prompt_tokens,
                queue=asyncio.Queue(),
            )
            self._requests[seq.seq_id] = request
            self._detokenizers[seq.seq_id] = IncrementalDetokenizer(
                self.output_tokenizer
            )
            self._wakeup.set()
            self._ensure_task()
            return request

    async def abort(self, seq_id: int) -> bool:
        async with self._lock:
            request = self._requests.get(seq_id)
            removed = self.engine.abort(seq_id)
            if request is not None and not request.finished:
                request.finished = True
                request.finish_reason = "abort"
                request.queue.put_nowait(
                    StreamEvent(
                        seq_id,
                        request.token_ids,
                        "",
                        True,
                        "abort",
                    )
                )
            return removed

    async def release(self, seq_id: int):
        """Drop bookkeeping after the consumer has read the terminal event."""
        async with self._lock:
            request = self._requests.get(seq_id)
            if request is not None and request.finished:
                self._requests.pop(seq_id, None)
                self._detokenizers.pop(seq_id, None)

    async def stream(self, request: AsyncRequest):
        async for event in request.stream():
            yield event
        await self.release(request.seq_id)

    async def collect(self, request: AsyncRequest) -> RequestOutput:
        last = None
        async for event in self.stream(request):
            last = event
        if last is None:
            raise RuntimeError("request stream ended without an event")
        return RequestOutput(
            request.seq_id,
            last.token_ids,
            last.finished,
            last.finish_reason,
        )

    async def generate(self, prompt, sampling_params: SampleParameter):
        request = await self.add_request(prompt, sampling_params)
        return await self.collect(request)

    def stats(self) -> dict:
        stats = self.engine.stats()
        stats["active_requests"] = len(self._requests)
        stats["max_waiting"] = self.max_waiting
        stats["last_error"] = str(self._error) if self._error is not None else None
        return stats

    async def _run_loop(self):
        try:
            while True:
                async with self._lock:
                    if self._stopping:
                        return
                    if self.engine.is_finished():
                        idle = True
                    else:
                        idle = False
                        _, _, updates = await asyncio.to_thread(self.engine.step_events)
                        self._dispatch(updates)
                if idle:
                    self._wakeup.clear()
                    # Recheck after clearing to close the add/wait race.
                    if self.engine.is_finished():
                        await self._wakeup.wait()
                    else:
                        continue
                else:
                    await asyncio.sleep(0)
        except BaseException as exc:  # surface failure to every active stream
            self._error = exc
            logger.exception("async engine step failed")
            async with self._lock:
                for request in self._requests.values():
                    if request.finished:
                        continue
                    request.finished = True
                    request.finish_reason = "abort"
                    request.queue.put_nowait(
                        StreamEvent(
                            request.seq_id,
                            request.token_ids,
                            "",
                            True,
                            "abort",
                        )
                    )

    def _dispatch(self, updates):
        for seq_id, token_ids, finished, finish_reason in updates:
            request = self._requests.get(seq_id)
            if request is None or request.finished:
                continue
            if not finished and token_ids == request.token_ids:
                continue
            detokenizer = self._detokenizers[seq_id]
            delta = detokenizer.update(token_ids, finished=finished)
            request.token_ids = token_ids
            request.text = detokenizer.text
            request.finished = finished
            request.finish_reason = finish_reason
            request.queue.put_nowait(
                StreamEvent(
                    seq_id,
                    token_ids,
                    delta,
                    finished,
                    finish_reason,
                )
            )

    async def stop(self):
        self._stopping = True
        self._wakeup.set()
        task = self._task
        if task is not None:
            await task
        async with self._lock:
            for seq_id, request in list(self._requests.items()):
                if not request.finished:
                    self.engine.abort(seq_id)
                    request.finished = True
                    request.finish_reason = "abort"
                    request.queue.put_nowait(
                        StreamEvent(
                            seq_id,
                            request.token_ids,
                            "",
                            True,
                            "abort",
                        )
                    )
            self._requests.clear()
            self._detokenizers.clear()
