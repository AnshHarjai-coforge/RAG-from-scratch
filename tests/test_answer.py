import io
import json
import urllib.error
from dataclasses import replace

import pytest

from rag.answer import FOUNDRY_TIMEOUT_SECONDS, generate_answer
from rag.config import get_settings
from rag.vector_store import RetrievalHit


def test_generate_answer_uses_the_stand_in_and_names_the_source():
    hit = RetrievalHit(
        id="fixed:expense.md:0",
        text="Submit a report within 30 calendar days.",
        distance=0.1,
        metadata={
            "source": "02_expense_reimbursement_policy.md",
            "section": "5. Submission requirements and deadlines",
        },
    )
    seen = {}

    def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return "30 calendar days"

    answer = generate_answer("When is it due?", [hit], get_settings(), complete=complete)

    assert answer == "30 calendar days"
    assert "If the context conflicts, say so." in seen["prompt"]
    assert (
        "02_expense_reimbursement_policy.md § 5. Submission requirements and deadlines"
        in seen["prompt"]
    )
    assert "30 calendar days" in seen["prompt"]


def test_generate_answer_returns_an_empty_stand_in(monkeypatch):
    def fail_urlopen(*args, **kwargs):
        raise AssertionError("complete() should not call Foundry")

    monkeypatch.setattr("rag.answer.urllib.request.urlopen", fail_urlopen)

    def complete(prompt: str) -> str:
        return ""

    answer = generate_answer("When is it due?", [], get_settings(), complete=complete)

    assert answer == ""


def test_missing_foundry_settings_name_the_variables():
    settings = replace(
        get_settings(),
        foundry_api_key=None,
        foundry_resource=None,
        foundry_model="",
    )

    with pytest.raises(RuntimeError) as error:
        generate_answer("When is it due?", [], settings)

    message = str(error.value)
    assert "ANTHROPIC_FOUNDRY_API_KEY" in message
    assert "ANTHROPIC_FOUNDRY_RESOURCE" in message
    assert "ANTHROPIC_MODEL" in message


def test_post_reads_the_text_blocks(monkeypatch):
    payload = json.dumps(
        {
            "content": [
                {"type": "text", "text": "Two days."},
                {"type": "thinking", "text": "skip"},
                {"type": "text", "text": "See the policy."},
            ]
        }
    ).encode()

    class FakeResponse:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        assert timeout == FOUNDRY_TIMEOUT_SECONDS
        assert request.get_header("Api-key") == "secret"
        assert request.get_header("X-api-key") == "secret"
        assert request.get_header("Anthropic-version") == "2023-06-01"
        assert b'"model": "claude-opus-5"' in request.data
        return FakeResponse()

    monkeypatch.setattr("rag.answer.urllib.request.urlopen", fake_urlopen)
    settings = replace(
        get_settings(),
        foundry_api_key="secret",
        foundry_resource="rg-test",
        foundry_model="claude-opus-5",
    )

    answer = generate_answer("How many remote days?", [], settings)

    assert answer == "Two days.\nSee the policy."


def test_post_rejects_thinking_and_tool_use(monkeypatch):
    payload = json.dumps(
        {
            "stop_reason": "max_tokens",
            "content": [
                {"type": "thinking", "thinking": "skip"},
                {"type": "tool_use", "name": "search"},
            ],
        }
    ).encode()

    class FakeResponse:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        assert timeout == FOUNDRY_TIMEOUT_SECONDS
        return FakeResponse()

    monkeypatch.setattr("rag.answer.urllib.request.urlopen", fake_urlopen)
    settings = replace(
        get_settings(),
        foundry_api_key="secret",
        foundry_resource="rg-test",
        foundry_model="claude-opus-5",
    )

    with pytest.raises(RuntimeError) as error:
        generate_answer("How many remote days?", [], settings)

    message = str(error.value)
    assert "thinking" in message
    assert "tool_use" in message
    assert "max_tokens" in message


def test_post_rejects_an_empty_content_list(monkeypatch):
    payload = json.dumps({"content": [], "stop_reason": "end_turn"}).encode()

    class FakeResponse:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        return FakeResponse()

    monkeypatch.setattr("rag.answer.urllib.request.urlopen", fake_urlopen)
    settings = replace(
        get_settings(),
        foundry_api_key="secret",
        foundry_resource="rg-test",
        foundry_model="claude-opus-5",
    )

    with pytest.raises(RuntimeError) as error:
        generate_answer("How many remote days?", [], settings)

    assert "end_turn" in str(error.value)


def test_post_reports_an_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"bad key"),
        )

    monkeypatch.setattr("rag.answer.urllib.request.urlopen", fake_urlopen)
    settings = replace(
        get_settings(),
        foundry_api_key="secret",
        foundry_resource="rg-test",
        foundry_model="claude-opus-5",
    )

    with pytest.raises(RuntimeError) as error:
        generate_answer("How many remote days?", [], settings)

    assert "401" in str(error.value)
    assert "bad key" in str(error.value)


def test_post_reports_a_timeout(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("rag.answer.urllib.request.urlopen", fake_urlopen)
    settings = replace(
        get_settings(),
        foundry_api_key="secret",
        foundry_resource="rg-test",
        foundry_model="claude-opus-5",
    )

    with pytest.raises(RuntimeError) as error:
        generate_answer("How many remote days?", [], settings)

    assert "120" in str(error.value)
