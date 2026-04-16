from __future__ import annotations

from types import SimpleNamespace

import sys

from app.ai import (
    ContextBlock,
    MockTextAIClient,
    OpenAITextAIClient,
    build_text_ai_client,
    stream_text_chunks,
)
from app.config import Settings


class StubCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            chunks = self.responses.pop(0)
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk))]
                    )
                    for chunk in chunks
                ]
            )
        content = self.responses.pop(0)
        choice = SimpleNamespace(message=SimpleNamespace(content=content))
        return SimpleNamespace(choices=[choice])


class StubEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])]
        )


class StubOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=StubCompletions(
                [
                    "Summary text",
                    [{"text": "Answer text"}],
                    ["Answer", " stream"],
                ]
            )
        )
        self.embeddings = StubEmbeddings()


def test_mock_ai_client_summary_answer_and_embeddings():
    client = MockTextAIClient()
    summary = client.summarize(
        title="report.pdf",
        media_type="pdf",
        text="Revenue improved. Margins also improved. Customer churn stayed low.",
    )
    answer = client.answer(
        question="What improved?",
        context=[
            ContextBlock(
                asset_id="1",
                filename="report.pdf",
                media_type="pdf",
                content="Revenue improved. Margins also improved.",
            )
        ],
    )
    embeddings = client.embed_texts(["Revenue improved", "Customer churn"])

    assert "report.pdf" in summary
    assert "improved" in answer.lower()
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 32


def test_build_text_ai_client_prefers_mock_without_key():
    settings = Settings(ai_provider="auto", openai_api_key=None)
    assert isinstance(build_text_ai_client(settings), MockTextAIClient)


def test_build_text_ai_client_uses_openai_when_key_present(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("app.ai.OpenAITextAIClient", lambda api_key, chat_model, embedding_model: sentinel)
    settings = Settings(ai_provider="auto", openai_api_key="token")

    assert build_text_ai_client(settings) is sentinel


def test_openai_text_ai_client_uses_stubbed_client():
    stub = StubOpenAIClient()
    client = OpenAITextAIClient(
        api_key="test-key",
        chat_model="gpt-test",
        embedding_model="embed-test",
        client=stub,
    )

    summary = client.summarize(title="clip.mp3", media_type="audio", text="Audio transcript")
    answer = client.answer(
        question="What happened?",
        context=[
            ContextBlock(
                asset_id="1",
                filename="clip.mp3",
                media_type="audio",
                content="A decision was made.",
                start_seconds=12.0,
                end_seconds=18.0,
            )
        ],
    )
    embeddings = client.embed_texts(["first", "second"])

    assert summary == "Summary text"
    assert answer == "Answer text"
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert stub.chat.completions.calls[1]["messages"][1]["content"].startswith("Question:")
    assert stub.embeddings.calls[0]["model"] == "embed-test"


def test_openai_text_ai_client_extract_text_handles_string_payload():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Plain string content"))]
    )
    assert OpenAITextAIClient._extract_text(response) == "Plain string content"


def test_openai_text_ai_client_streams_answer_chunks():
    stub = SimpleNamespace(
        chat=SimpleNamespace(completions=StubCompletions([[[{"text": "Answer"}, {"text": " stream"}]]])),
        embeddings=StubEmbeddings(),
    )
    client = OpenAITextAIClient(
        api_key="test-key",
        chat_model="gpt-test",
        embedding_model="embed-test",
        client=stub,
    )

    chunks = list(
        client.stream_answer(
            question="What happened?",
            context=[
                ContextBlock(
                    asset_id="1",
                    filename="clip.mp3",
                    media_type="audio",
                    content="A decision was made.",
                )
            ],
        )
    )

    assert "".join(chunks) == "Answer stream"


def test_stream_text_chunks_and_mock_fallback_paths():
    assert list(stream_text_chunks("", chunk_size=2)) == []
    assert list(stream_text_chunks("one two three", chunk_size=2)) == ["one two", " three"]

    client = MockTextAIClient()
    assert client.summarize(title="empty.pdf", media_type="pdf", text="   ") == "No extractable text was found in empty.pdf."
    assert client.answer(question="Anything?", context=[]) == "I could not find relevant context in the uploaded files."
    assert client.answer(
        question="Anything?",
        context=[ContextBlock(asset_id="1", filename="blank.txt", media_type="pdf", content="   ")],
    ) == ""


def test_openai_client_handles_empty_inputs_and_constructs_default_client(monkeypatch):
    client = OpenAITextAIClient(
        api_key="test-key",
        chat_model="gpt-test",
        embedding_model="embed-test",
        client=StubOpenAIClient(),
    )
    assert client.summarize(title="clip.mp3", media_type="audio", text="   ") == "No extractable text was found in clip.mp3."
    assert client.answer(question="Anything?", context=[]) == "I could not find relevant context in the uploaded files."
    assert list(client.stream_answer(question="Anything?", context=[])) == [
        "I could not find relevant context in the uploaded files."
    ]
    assert client.embed_texts([]) == []

    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key: str):
            captured["api_key"] = api_key

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    created = OpenAITextAIClient(api_key="real-key", chat_model="gpt-test", embedding_model="embed-test")
    assert captured["api_key"] == "real-key"
    assert created.client is not None
