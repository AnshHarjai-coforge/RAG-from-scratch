"""Ask Claude on Microsoft Foundry to answer from retrieved chunks."""

import json
import urllib.error
import urllib.request

from rag.config import Settings
from rag.vector_store import RetrievalHit

ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 1024
FOUNDRY_TIMEOUT_SECONDS = 120


def build_prompt(question: str, hits: list[RetrievalHit]) -> str:
    """Number the chunks and place the question after them."""

    blocks = []
    for index, hit in enumerate(hits, start=1):
        source = hit.metadata.get("source", "unknown")
        section = hit.metadata.get("section") or "body"
        blocks.append(f"[{index}] ({source} § {section})\n{hit.text}")
    context = "\n\n".join(blocks)
    return (
        "Answer using only the context. If the context conflicts, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def generate_answer(
    question: str,
    hits: list[RetrievalHit],
    settings: Settings,
    complete=None,
) -> str:
    """Return an answer written from these chunks.

    ``complete`` is a test stand-in. When it is omitted, the prompt is sent
    to Claude on Foundry.
    """

    prompt = build_prompt(question, hits)
    if complete is not None:
        return complete(prompt)
    _require_foundry(settings)
    return _post(prompt, settings)


def _require_foundry(settings: Settings) -> None:
    """Raise if the Foundry key, resource, or model is missing."""

    missing = []
    if not settings.foundry_api_key:
        missing.append("ANTHROPIC_FOUNDRY_API_KEY")
    if not settings.foundry_resource:
        missing.append("ANTHROPIC_FOUNDRY_RESOURCE")
    if not settings.foundry_model:
        missing.append("ANTHROPIC_MODEL")
    if missing:
        raise RuntimeError("Set " + ", ".join(missing))


def _post(prompt: str, settings: Settings) -> str:
    """POST one Messages request and return the text blocks.

    Raise if the body has no text, or if the request exceeds the time limit.
    """

    url = f"https://{settings.foundry_resource}.services.ai.azure.com/anthropic/v1/messages"
    body = json.dumps(
        {
            "model": settings.foundry_model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "content-type": "application/json",
            "api-key": settings.foundry_api_key or "",
            "x-api-key": settings.foundry_api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=FOUNDRY_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Foundry request failed ({error.code}): {detail}") from error
    except TimeoutError as error:
        raise RuntimeError(
            f"Foundry request timed out after {FOUNDRY_TIMEOUT_SECONDS} seconds"
        ) from error

    blocks = payload.get("content", [])
    parts = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        types = [block.get("type") for block in blocks]
        stop_reason = payload.get("stop_reason")
        raise RuntimeError(
            f"Foundry response had no text (types: {types}; stop_reason: {stop_reason})"
        )
    return text
