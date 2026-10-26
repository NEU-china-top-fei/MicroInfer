"""OpenAI-compatible FastAPI frontend for MicroInfer."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from microinfer.config import SampleParameter
from microinfer.engine.async_engine import AsyncEngine, BackpressureError
from microinfer.engine.contracts import RequestOutput
from microinfer.serve.metrics import ServiceMetrics


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: Any
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    stream: bool = False
    stop: Any = None
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    stream: bool = False
    stop: Any = None
    ignore_eos: bool = False


class StopBuffer:
    """Hold back a suffix so a stop string is never emitted partially."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.keep = max((len(s) for s in self.stops), default=1) - 1
        self.pending = ""

    def push(self, delta: str, *, finished: bool = False):
        self.pending += delta
        for stop in self.stops:
            index = self.pending.find(stop)
            if index >= 0:
                emit = self.pending[:index]
                self.pending = ""
                return emit, True
        if finished:
            emit, self.pending = self.pending, ""
            return emit, False
        if len(self.pending) <= self.keep:
            return "", False
        if self.keep == 0:
            emit, self.pending = self.pending, ""
            return emit, False
        emit = self.pending[: -self.keep]
        self.pending = self.pending[-self.keep :]
        return emit, False


def _sampling(request) -> SampleParameter:
    greedy = request.temperature <= 1e-8
    temperature = 1.0 if greedy else float(request.temperature)
    return SampleParameter(
        temperature=temperature,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
        greedy=greedy,
    )


def _stops(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError("stop must be a string or a list of strings")


def _normalize_prompts(prompt) -> list[str | list[int]]:
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if not prompt:
            raise ValueError("prompt must not be empty")
        if all(type(token) is int for token in prompt):
            return [prompt]
        if all(
            isinstance(item, str)
            or (
                isinstance(item, list)
                and item
                and all(type(token) is int for token in item)
            )
            for item in prompt
        ):
            return list(prompt)
    raise ValueError("prompt must be a string, token list, or list of prompts")


def _chat_prompt(tokenizer, messages) -> list[int]:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        raise ValueError("tokenizer does not provide a chat template")
    tokens = apply(
        [message.model_dump() for message in messages],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if not isinstance(tokens, list):
        raise TypeError("chat template did not return token IDs")
    return list(tokens)


def _completion_payload(request_id, model, choices, usage=None):
    payload = {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _chat_payload(request_id, model, choices, usage=None):
    payload = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _finish_reason(reason):
    return {"length": "length", "eos": "stop", "stop": "stop", "abort": "stop"}.get(
        reason, "stop"
    )


def create_app(llm, *, model_name: str = "microinfer", max_waiting: int = 128):
    async_engine = AsyncEngine(llm, max_waiting=max_waiting)
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await async_engine.stop()

    app = FastAPI(title="MicroInfer", version="0.1.0", lifespan=lifespan)
    app.state.async_engine = async_engine
    app.state.metrics = metrics
    app.state.model_name = model_name
    tokenizer = llm.output_processor.tokenizer

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name, **async_engine.stats()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus():
        return metrics.render(async_engine.stats())

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "microinfer",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        try:
            prompts = _normalize_prompts(request.prompt)
            stops = _stops(request.stop)
            params = _sampling(request)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.stream and len(prompts) != 1:
            raise HTTPException(
                status_code=400, detail="streaming supports one prompt per request"
            )
        metrics.inc("microinfer_requests_total")
        request_id = f"cmpl-{uuid.uuid4().hex}"
        if request.stream:
            metrics.inc("microinfer_stream_requests_total")
            try:
                handle = await async_engine.add_request(prompts[0], params)
            except BackpressureError as exc:
                metrics.inc("microinfer_requests_failed_total")
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            except (ValueError, TypeError) as exc:
                metrics.inc("microinfer_requests_failed_total")
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return StreamingResponse(
                _stream_completion(
                    async_engine,
                    metrics,
                    handle,
                    stops,
                    request_id,
                    model_name,
                    chat=False,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            results = await asyncio.gather(
                *[_run_one(async_engine, metrics, prompt, params) for prompt in prompts]
            )
        except BackpressureError as exc:
            metrics.inc("microinfer_requests_failed_total")
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            metrics.inc("microinfer_requests_failed_total")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for index, (output, handle) in enumerate(results):
            text, stopped = _truncate_stop(
                tokenizer.decode(list(output.token_ids), skip_special_tokens=True),
                stops,
            )
            choices.append(
                {
                    "text": text,
                    "index": index,
                    "logprobs": None,
                    "finish_reason": "stop"
                    if stopped
                    else _finish_reason(output.finish_reason),
                }
            )
            prompt_tokens += handle.prompt_tokens
            completion_tokens += len(output.token_ids)
            metrics.inc("microinfer_requests_succeeded_total")
            metrics.inc("microinfer_completion_tokens_total", len(output.token_ids))
        metrics.inc("microinfer_prompt_tokens_total", prompt_tokens)
        return _completion_payload(
            request_id,
            model_name,
            choices,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        try:
            tokens = _chat_prompt(tokenizer, request.messages)
            stops = _stops(request.stop)
            params = _sampling(request)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        metrics.inc("microinfer_requests_total")
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        if request.stream:
            metrics.inc("microinfer_stream_requests_total")
            try:
                handle = await async_engine.add_request(tokens, params)
            except BackpressureError as exc:
                metrics.inc("microinfer_requests_failed_total")
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            except (ValueError, TypeError) as exc:
                metrics.inc("microinfer_requests_failed_total")
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return StreamingResponse(
                _stream_completion(
                    async_engine,
                    metrics,
                    handle,
                    stops,
                    request_id,
                    model_name,
                    chat=True,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            output, handle = await _run_one(async_engine, metrics, tokens, params)
        except BackpressureError as exc:
            metrics.inc("microinfer_requests_failed_total")
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            metrics.inc("microinfer_requests_failed_total")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        text, stopped = _truncate_stop(
            tokenizer.decode(list(output.token_ids), skip_special_tokens=True),
            stops,
        )
        metrics.inc("microinfer_prompt_tokens_total", handle.prompt_tokens)
        metrics.inc("microinfer_completion_tokens_total", len(output.token_ids))
        metrics.inc(
            "microinfer_requests_aborted_total"
            if output.finish_reason == "abort"
            else "microinfer_requests_succeeded_total"
        )
        return _chat_payload(
            request_id,
            model_name,
            [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                    if stopped
                    else _finish_reason(output.finish_reason),
                }
            ],
            {
                "prompt_tokens": handle.prompt_tokens,
                "completion_tokens": len(output.token_ids),
                "total_tokens": handle.prompt_tokens + len(output.token_ids),
            },
        )

    return app


async def _run_one(async_engine, metrics, prompt, params):
    handle = await async_engine.add_request(prompt, params)
    first_at = None
    last_at = None
    first_tokens = 0
    async for event in async_engine.stream(handle):
        if not event.token_ids:
            continue
        now = time.perf_counter()
        if first_at is None:
            first_at = now
            first_tokens = len(event.token_ids)
            metrics.ttft.observe(now - handle.created_at)
        last_at = now
    if first_at is not None and last_at is not None:
        tokens = max(1, len(handle.token_ids) - first_tokens)
        metrics.tpot.observe((last_at - first_at) / tokens)
    output = RequestOutput(
        handle.seq_id,
        handle.token_ids,
        handle.finished,
        handle.finish_reason,
    )
    return output, handle


def _truncate_stop(text: str, stops: list[str]):
    positions = [text.find(stop) for stop in stops if stop in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


def _sse(payload) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_completion(
    async_engine,
    metrics,
    handle,
    stops,
    request_id,
    model_name,
    *,
    chat: bool,
):
    object_name = "chat.completion.chunk" if chat else "text_completion"
    stop_buffer = StopBuffer(stops)
    first_at = None
    last_at = None
    first_tokens = 0
    completed = False
    metrics.inc("microinfer_prompt_tokens_total", handle.prompt_tokens)
    try:
        async for event in async_engine.stream(handle):
            now = time.perf_counter()
            if first_at is None and event.token_ids:
                first_at = now
                first_tokens = len(event.token_ids)
                metrics.ttft.observe(now - handle.created_at)
            if event.token_ids:
                last_at = now
            emit, stopped = stop_buffer.push(event.text_delta, finished=event.finished)
            if stopped:
                await async_engine.abort(handle.seq_id)
                await async_engine.release(handle.seq_id)
                completed = True
                metrics.inc("microinfer_requests_succeeded_total")
                metrics.inc("microinfer_completion_tokens_total", len(event.token_ids))
                yield _sse(
                    _stream_payload(
                        request_id,
                        model_name,
                        object_name,
                        emit,
                        "stop",
                        chat,
                    )
                )
                yield "data: [DONE]\n\n"
                return
            if event.finished:
                completed = True
                await async_engine.release(handle.seq_id)
                metrics.inc(
                    "microinfer_requests_aborted_total"
                    if event.finish_reason == "abort"
                    else "microinfer_requests_succeeded_total"
                )
                metrics.inc("microinfer_completion_tokens_total", len(event.token_ids))
                if last_at is not None and first_at is not None:
                    tokens = max(1, len(event.token_ids) - first_tokens)
                    metrics.tpot.observe((last_at - first_at) / tokens)
                yield _sse(
                    _stream_payload(
                        request_id,
                        model_name,
                        object_name,
                        emit,
                        _finish_reason(event.finish_reason),
                        chat,
                    )
                )
                yield "data: [DONE]\n\n"
                return
            if emit:
                yield _sse(
                    _stream_payload(
                        request_id,
                        model_name,
                        object_name,
                        emit,
                        None,
                        chat,
                    )
                )
    finally:
        if not completed:
            await async_engine.abort(handle.seq_id)
            await async_engine.release(handle.seq_id)


def _stream_payload(request_id, model, object_name, text, finish_reason, chat):
    if chat:
        return {
            "id": request_id,
            "object": object_name,
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": finish_reason,
                }
            ],
        }
    return {
        "id": request_id,
        "object": object_name,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}
        ],
    }
