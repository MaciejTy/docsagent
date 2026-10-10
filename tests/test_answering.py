from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from answering import DEFAULT_MODEL, SYSTEM_PROMPT, AnswerError, build_prompt, generate_answer, get_client
from models import SearchResult

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def result(number, source="docs/a.md", heading="A > B", text="body"):
    return SearchResult(chunk_id=number, source=source, heading_path=heading, text=text, score=0.5)


def reply(*texts, stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking="")]
    content += [SimpleNamespace(type="text", text=text) for text in texts]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeClient:
    """Udaje anthropic.Anthropic: zapamiętuje argumenty i zwraca gotową odpowiedź albo rzuca błąd."""

    def __init__(self, response=None, error=None):
        self.messages = self
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def status_error(cls, status):
    return cls("boom sk-secret", response=httpx2.Response(status, request=REQUEST), body=None)


def test_fragments_are_numbered_in_result_order():
    prompt = build_prompt("Q?", [result(7, text="first"), result(3, source="docs/b.md", heading="C", text="second")])

    assert '<fragment number="1" source="docs/a.md" section="A > B">\nfirst\n</fragment>' in prompt
    assert '<fragment number="2" source="docs/b.md" section="C">\nsecond\n</fragment>' in prompt
    assert prompt.index("first") < prompt.index("second")


def test_section_is_omitted_for_empty_heading_path():
    prompt = build_prompt("Q?", [result(1, heading="")])

    assert '<fragment number="1" source="docs/a.md">' in prompt
    assert "section=" not in prompt


def test_attributes_are_escaped():
    prompt = build_prompt("Q?", [result(1, source='we"ird<.md', heading='Use "x" <now>')])

    assert 'source="we&quot;ird&lt;.md"' in prompt
    assert 'section="Use &quot;x&quot; &lt;now>"' in prompt


def test_question_comes_after_fragments():
    prompt = build_prompt("How do I run it?", [result(1)])

    assert prompt.endswith("<question>\nHow do I run it?\n</question>")
    assert prompt.index("</fragments>") < prompt.index("<question>")


def test_generate_answer_returns_text_and_sends_expected_request(monkeypatch):
    monkeypatch.delenv("DOCSAGENT_MODEL", raising=False)
    client = FakeClient(reply("Run `fastapi dev` ", "[1]."))
    results = [result(1)]

    assert generate_answer("How?", results, client=client) == "Run `fastapi dev` [1]."
    assert client.kwargs["model"] == DEFAULT_MODEL == "claude-opus-5-5"
    assert client.kwargs["system"] == SYSTEM_PROMPT
    assert client.kwargs["max_tokens"] == 16000
    assert client.kwargs["output_config"] == {"effort": "low"}
    assert client.kwargs["messages"] == [{"role": "user", "content": build_prompt("How?", results)}]


def test_model_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("DOCSAGENT_MODEL", "claude-haiku-5-5")
    client = FakeClient(reply("ok"))

    generate_answer("How?", [result(1)], client=client)

    assert client.kwargs["model"] == "claude-haiku-5-5"


def test_truncated_answer_is_still_returned():
    client = FakeClient(reply("partial answer", stop_reason="max_tokens"))

    assert generate_answer("How?", [result(1)], client=client) == "partial answer"


def test_refusal_is_an_error():
    client = FakeClient(reply("I cannot help with that.", stop_reason="refusal"))

    with pytest.raises(AnswerError, match="declined"):
        generate_answer("How?", [result(1)], client=client)


def test_empty_answer_is_an_error():
    with pytest.raises(AnswerError, match="empty"):
        generate_answer("How?", [result(1)], client=FakeClient(reply()))
    with pytest.raises(AnswerError, match="empty"):
        generate_answer("How?", [result(1)], client=FakeClient(reply("   \n")))


def test_missing_credentials_are_reported():
    error = TypeError("Could not resolve authentication method. Expected one of api_key, auth_token...")

    with pytest.raises(AnswerError, match="ANTHROPIC_API_KEY"):
        generate_answer("How?", [result(1)], client=FakeClient(error=error))


@pytest.mark.parametrize("error, message", [
    (status_error(anthropic.AuthenticationError, 401), "rejected the credentials"),
    (status_error(anthropic.RateLimitError, 429), "rate limited"),
    (status_error(anthropic.InternalServerError, 500), "HTTP 500"),
    (anthropic.APITimeoutError(request=REQUEST), "60 seconds"),
    (anthropic.APIConnectionError(request=REQUEST), "Could not connect"),
])
def test_sdk_errors_become_answer_errors(error, message):
    with pytest.raises(AnswerError, match=message) as raised:
        generate_answer("How?", [result(1)], client=FakeClient(error=error))
    assert "sk-secret" not in str(raised.value)


def test_unexpected_error_does_not_leak_details():
    client = FakeClient(error=RuntimeError("secret internal detail"))

    with pytest.raises(AnswerError) as raised:
        generate_answer("How?", [result(1)], client=client)
    assert "RuntimeError" in str(raised.value)
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize("response", [
    "<html>proxy login page</html>",
    SimpleNamespace(stop_reason="end_turn", content=None),
])
def test_malformed_response_is_an_error(response):
    with pytest.raises(AnswerError, match="Unexpected error"):
        generate_answer("How?", [result(1)], client=FakeClient(response))


def test_failure_details_are_logged_for_the_developer(caplog):
    client = FakeClient(error=RuntimeError("secret internal detail"))

    with pytest.raises(AnswerError):
        generate_answer("How?", [result(1)], client=client)
    assert "secret internal detail" in caplog.text


def test_client_fails_fast_instead_of_retrying(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_client.cache_clear()
    try:
        client = get_client()
        assert client.max_retries == 0
        assert client.timeout.read == 60.0
        assert client.timeout.connect == 5.0
    finally:
        get_client.cache_clear()
