from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterator
from typing import Protocol, Sequence

from app.config import Settings
from app.utils import truncate_text


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def _tokens(text: str) -> set[str]:
    return {item for item in re.findall(r"[a-z0-9]+", text.lower()) if len(item) > 2}


def _hash_embedding(text: str, dimensions: int = 32) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(dimensions):
            vector[index] += (digest[index] / 255.0) - 0.5

    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


def stream_text_chunks(text: str, *, chunk_size: int = 18) -> Iterator[str]:
    words = text.split()
    if not words:
        return

    for start in range(0, len(words), chunk_size):
        prefix = "" if start == 0 else " "
        yield prefix + " ".join(words[start : start + chunk_size])


@dataclass(slots=True)
class ContextBlock:
    asset_id: str
    filename: str
    media_type: str
    content: str
    start_seconds: float | None = None
    end_seconds: float | None = None


class TextAIClient(Protocol):
    def summarize(self, *, title: str, media_type: str, text: str) -> str:
        ...

    def answer(self, *, question: str, context: Sequence[ContextBlock]) -> str:
        ...

    def stream_answer(self, *, question: str, context: Sequence[ContextBlock]) -> Iterator[str]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class MockTextAIClient:
    def summarize(self, *, title: str, media_type: str, text: str) -> str:
        if not text.strip():
            return f"No extractable text was found in {title}."
        summary_source = _sentences(text)[:3]
        summary = " ".join(summary_source) if summary_source else text.strip()
        return truncate_text(f"{media_type.upper()} summary for {title}: {summary}", 320)

    def answer(self, *, question: str, context: Sequence[ContextBlock]) -> str:
        if not context:
            return "I could not find relevant context in the uploaded files."

        question_tokens = _tokens(question)
        candidates: list[tuple[int, str]] = []

        for block in context:
            for sentence in _sentences(block.content) or [block.content]:
                score = len(question_tokens & _tokens(sentence))
                candidates.append((score, sentence.strip()))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_sentences = [item[1] for item in candidates[:2] if item[1]]
        if not best_sentences:
            best_sentences = [context[0].content.strip()]

        return " ".join(best_sentences)

    def stream_answer(self, *, question: str, context: Sequence[ContextBlock]) -> Iterator[str]:
        yield from stream_text_chunks(self.answer(question=question, context=context))

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_embedding(text) for text in texts]


class OpenAITextAIClient:
    def __init__(self, api_key: str, chat_model: str, embedding_model: str, client=None):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client

    def summarize(self, *, title: str, media_type: str, text: str) -> str:
        if not text.strip():
            return f"No extractable text was found in {title}."

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize user-provided content. Keep the summary factual, concise, and useful."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize this {media_type} named {title} in 4-6 bullet-free sentences.\n\n{text[:12000]}"
                    ),
                },
            ],
        )
        return self._extract_text(response)

    def answer(self, *, question: str, context: Sequence[ContextBlock]) -> str:
        if not context:
            return "I could not find relevant context in the uploaded files."

        rendered_context = []
        for index, block in enumerate(context, start=1):
            prefix = f"[{index}] {block.filename} ({block.media_type})"
            if block.start_seconds is not None:
                prefix += f" [{block.start_seconds:.2f}s - {block.end_seconds or block.start_seconds:.2f}s]"
            rendered_context.append(f"{prefix}\n{block.content}")

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied context. If the answer is not present, say so plainly. "
                        "Keep the answer concise and helpful."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nContext:\n\n" + "\n\n".join(rendered_context)
                    ),
                },
            ],
        )
        return self._extract_text(response)

    def stream_answer(self, *, question: str, context: Sequence[ContextBlock]) -> Iterator[str]:
        if not context:
            yield "I could not find relevant context in the uploaded files."
            return

        rendered_context = []
        for index, block in enumerate(context, start=1):
            prefix = f"[{index}] {block.filename} ({block.media_type})"
            if block.start_seconds is not None:
                prefix += f" [{block.start_seconds:.2f}s - {block.end_seconds or block.start_seconds:.2f}s]"
            rendered_context.append(f"{prefix}\n{block.content}")

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.1,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied context. If the answer is not present, say so plainly. "
                        "Keep the answer concise and helpful."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nContext:\n\n" + "\n\n".join(rendered_context)
                    ),
                },
            ],
        )

        for event in response:
            delta = event.choices[0].delta
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                yield content
                continue

            for item in content or []:
                text = getattr(item, "text", None) or item.get("text")
                if text:
                    yield text

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.embedding_model, input=list(texts))
        return [item.embedding for item in response.data]

    @staticmethod
    def _extract_text(response) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        parts = []
        for item in content:
            text = getattr(item, "text", None) or item.get("text")
            if text:
                parts.append(text)
        return "\n".join(parts).strip()


class BARTTextAIClient:
    """Local BART-based summarization with mock Q&A and embeddings."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from transformers import pipeline

            self.summarizer = pipeline("summarization", model=self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load BART model '{self.model_name}'. "
                "Ensure transformers and torch are installed: "
                "pip install transformers torch"
            ) from e

    def summarize(self, *, title: str, media_type: str, text: str) -> str:
        if not text.strip():
            return f"No extractable text was found in {title}."

        text_to_summarize = text[:1024]  # BART has token limits, cap input
        try:
            summary_result = self.summarizer(text_to_summarize, max_length=150, min_length=30, do_sample=False)
            if summary_result and isinstance(summary_result, list):
                summary_text = summary_result[0].get("summary_text", "").strip()
                return truncate_text(
                    f"{media_type.upper()} summary for {title}: {summary_text}",
                    320,
                )
            return truncate_text(f"{media_type.upper()} summary for {title}: {text_to_summarize[:300]}", 320)
        except Exception:
            summary_source = _sentences(text)[:3]
            summary = " ".join(summary_source) if summary_source else text.strip()
            return truncate_text(f"{media_type.upper()} summary for {title}: {summary}", 320)

    def answer(self, *, question: str, context: Sequence[ContextBlock]) -> str:
        if not context:
            return "I could not find relevant context in the uploaded files."

        question_tokens = _tokens(question)
        candidates: list[tuple[int, str]] = []

        for block in context:
            for sentence in _sentences(block.content) or [block.content]:
                score = len(question_tokens & _tokens(sentence))
                candidates.append((score, sentence.strip()))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_sentences = [item[1] for item in candidates[:2] if item[1]]
        if not best_sentences:
            best_sentences = [context[0].content.strip()]

        return " ".join(best_sentences)

    def stream_answer(self, *, question: str, context: Sequence[ContextBlock]) -> Iterator[str]:
        yield from stream_text_chunks(self.answer(question=question, context=context))

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_embedding(text) for text in texts]


def build_text_ai_client(settings: Settings) -> TextAIClient:
    provider = settings.ai_provider.lower()
    if provider == "mock":
        return MockTextAIClient()
    if settings.use_bart_for_summarization:
        return BARTTextAIClient(model_name=settings.bart_summarization_model)
    if settings.openai_api_key:
        return OpenAITextAIClient(
            api_key=settings.openai_api_key,
            chat_model=settings.openai_chat_model,
            embedding_model=settings.openai_embedding_model,
        )
    return MockTextAIClient()
