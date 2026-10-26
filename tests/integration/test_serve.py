import json

import pytest

from microinfer.config import QwenConfig
from microinfer.config.engine import EngineConfig
from microinfer.llm import LLM
from microinfer.model.qwen3 import QwenModel

pytestmark = pytest.mark.integration


class Tokenizer:
    eos_token_id = None

    def decode(self, ids, **kwargs):
        return "".join(str(i % 10) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3]


def _app():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from microinfer.serve.app import create_app

    model = QwenModel(
        QwenConfig(
            vocab_size=16,
            hidden_dim=8,
            MLP_inter_dim=16,
            num_q_head=2,
            num_kv_head=1,
            head_dim=4,
            num_hidden_layer=1,
            max_pos_embed=32,
        )
    )
    llm = LLM.from_model(
        model,
        Tokenizer(),
        block_size=2,
        num_blocks=16,
        engine_config=EngineConfig(
            max_model_len=16,
            max_num_batched_tokens=8,
            max_num_seqs=4,
        ),
    )
    return TestClient(create_app(llm, model_name="tiny", max_waiting=4))


def test_stop_buffer_never_emits_partial_stop():
    pytest.importorskip("fastapi")
    from microinfer.serve.app import StopBuffer

    buffer = StopBuffer(["STOP"])
    assert buffer.push("hello ST") == ("hello", False)
    assert buffer.push("OPtail") == (" ", True)


def test_openai_endpoints_stream_and_metrics():
    with _app() as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"

        response = client.post(
            "/v1/completions",
            json={
                "model": "tiny",
                "prompt": [1, 2],
                "max_tokens": 2,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"]["completion_tokens"] == 2

        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        assert chat.status_code == 200
        assert chat.json()["choices"][0]["message"]["role"] == "assistant"

        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "tiny",
                "prompt": [1, 2],
                "max_tokens": 2,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            },
        ) as stream:
            lines = list(stream.iter_lines())
        chunks = [
            json.loads(line[len("data: ") :])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert chunks and chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert [line for line in lines if line][-1] == "data: [DONE]"

        metrics = client.get("/metrics").text
        assert "microinfer_requests_total" in metrics
        assert "microinfer_ttft_seconds_count" in metrics
        assert "microinfer_kv_utilization" in metrics
