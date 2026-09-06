"""`transcribe()` always returns text. The structured extras are optional -- and gated.

Transposed from `unified-library-ts/src/helpers/transcribe.ts` and
`src/llm/providers/openai/transcription.ts`.

Segments, word timings and language detection are not universally available, and
on OpenAI each is gated to a DIFFERENT model. Returning empty lists for all of
them would make "this model cannot do it" indistinguishable from "there were no
words", which is the difference between a limitation and a result.

So the extras are None when unavailable and a list when supported, and asking
for one that cannot work WARNS rather than being silently dropped -- a caller
who asked for speakers must learn they are not coming, instead of quietly
receiving plain text.

Three response shapes come back from one OpenAI endpoint, selected by
`response_format` and constrained by the model. No model returns everything:
speaker labels and word timings live on different ones, which is why
`diarization` and word timestamps are mutually exclusive rather than two flags a
caller may combine.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: What a generateContent provider is asked, when STT is a normal completion.
DEFAULT_TRANSCRIBE_PROMPT = "Transcribe this audio exactly. Return only the transcript."

#: Options only the OpenAI transcription models can deliver.
STRUCTURED_ONLY = ("languages", "keywords", "timestamps", "diarization")

_BASE_URL = "https://api.openai.com"


@dataclass(frozen=True)
class TranscriptWord:
    """One word, timed in seconds from the start of the audio."""

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptSegment:
    """A timed run of transcript text.

    `speaker` is present only when diarization ran. No provider surface returns
    speakers and word timings together, so a segment carries whichever the
    chosen model produces.
    """

    start: float
    end: float
    text: str = ""
    #: Whisper numbers segments from 0, so `id: 0` is real and must survive.
    id: str | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """The transcript, plus whatever structure the model could supply.

    Every extra is None rather than `[]` when the model does not produce it: an
    empty list says "there were none", which is a different answer from "this
    model cannot tell you".
    """

    text: str
    #: The single language the model reported, when it reports one.
    language: str | None = None
    #: Candidate languages, on the models that report several.
    languages: Sequence[str] | None = None
    segments: Sequence[TranscriptSegment] | None = None
    words: Sequence[TranscriptWord] | None = None
    duration_seconds: float | None = None
    model: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


def parse_transcription(body: Any) -> TranscriptionResult:
    """One response body -> the unified result, dropping nothing it carried."""
    raw: Mapping[str, Any] = body if isinstance(body, Mapping) else {}

    languages = [
        str(entry["code"])
        for entry in (raw.get("languages") or [])
        if isinstance(entry, Mapping) and isinstance(entry.get("code"), str)
    ]
    segments = [
        TranscriptSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            text=str(s.get("text") or ""),
            id=None if s.get("id") is None else str(s["id"]),
            speaker=s.get("speaker"),
        )
        for s in (raw.get("segments") or [])
        if isinstance(s, Mapping)
        and isinstance(s.get("start"), (int, float))
        and isinstance(s.get("end"), (int, float))
    ]
    words = [
        TranscriptWord(word=str(w["word"]), start=float(w["start"]), end=float(w["end"]))
        for w in (raw.get("words") or [])
        if isinstance(w, Mapping)
        and isinstance(w.get("word"), str)
        and isinstance(w.get("start"), (int, float))
        and isinstance(w.get("end"), (int, float))
    ]

    return TranscriptionResult(
        text=str(raw.get("text") or ""),
        language=raw.get("language") if isinstance(raw.get("language"), str) else None,
        # Empty stays None: see the class docstring.
        languages=tuple(languages) or None,
        segments=tuple(segments) or None,
        words=tuple(words) or None,
        duration_seconds=_duration_of(raw),
        raw=raw,
    )


def _duration_of(raw: Mapping[str, Any]) -> float | None:
    """`verbose_json` reports it at the top level; the token-billed models only
    inside a duration-typed usage object."""
    duration = raw.get("duration")
    if isinstance(duration, (int, float)):
        return float(duration)
    usage = raw.get("usage")
    if isinstance(usage, Mapping) and usage.get("type") == "duration":
        seconds = usage.get("seconds")
        if isinstance(seconds, (int, float)):
            return float(seconds)
    return None


def _filename_for(mime_type: str) -> str:
    """A filename with a supported extension -- the endpoint requires one."""
    for needle, name in (
        ("mpeg", "audio.mp3"),
        ("mp3", "audio.mp3"),
        ("mp4", "audio.m4a"),
        ("m4a", "audio.m4a"),
        ("ogg", "audio.ogg"),
        ("flac", "audio.flac"),
        ("webm", "audio.webm"),
    ):
        if needle in mime_type:
            return name
    return "audio.wav"


def _load_audio(audio: Any, mime_type: str | None = None) -> tuple[bytes, str]:
    """Bytes and a MIME type, from whichever of the three shapes was given.

    The same `str | Path | bytes` union `attachments=` takes, so there is one
    rule for "how do I hand this library a file" across the whole surface.
    """
    from .helpers.content import load_bytes

    if isinstance(audio, Mapping):
        return _load_audio(audio.get("data"), audio.get("mimeType") or mime_type)
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio), mime_type or "audio/wav"
    data, detected = load_bytes(audio if isinstance(audio, (str, Path)) else str(audio))
    return data, mime_type or detected


def transcribe(
    *,
    model: str,
    audio: Any,
    api_key: str | None = None,
    provider: str | None = None,
    language: str | None = None,
    languages: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
    timestamps: str | None = None,
    diarization: bool = False,
    prompt: str | None = None,
    mime_type: str | None = None,
    engine: Any = None,
    transport: Any = None,
    base_url: str | None = None,
) -> TranscriptionResult:
    """Audio in, transcript out.

    `timestamps="word"` asks for segment and word timings; `diarization=True`
    asks for speaker labels. They select different response formats and no model
    serves both, so asking for both is refused here rather than by a 400.
    """
    from .helpers.client_resolver import is_namespaced_model_id, parse_model_id

    provider_name, model_name = (
        parse_model_id(model) if is_namespaced_model_id(model) else (provider, model)
    )
    if not provider_name:
        raise ValueError(
            'transcribe: name the provider, either as `provider=` or as "provider/model".'
        )
    if timestamps and diarization:
        raise ValueError(
            "transcribe: timestamps and diarization select different response formats "
            "and no model serves both. Ask for one."
        )

    key = api_key or (engine.api_keys.get(provider_name) if engine is not None else None)
    if not key:
        raise ValueError(
            f'transcribe: no API key for provider "{provider_name}". Pass api_key= or '
            f"engine.api_keys."
        )

    # The dedicated endpoint is for a dedicated model. Routing on the PROVIDER
    # instead sent OpenAI's default chat model to `/v1/audio/transcriptions`,
    # which answers 404 -- and a chat model asked to listen is scenario 15's
    # question, which every provider answers through a normal completion.
    if not _is_transcription_model(provider_name, model_name, engine):
        return _via_completion(
            model=model,
            provider=provider_name,
            audio=audio,
            api_key=key,
            prompt=prompt,
            engine=engine,
            transport=transport,
            requested=_structured_asked_for(languages, keywords, timestamps, diarization),
        )

    data, detected = _load_audio(audio, mime_type)
    result = _openai_transcribe(
        data=data,
        mime_type=detected,
        model=model_name,
        api_key=key,
        language=language,
        languages=languages,
        keywords=keywords,
        timestamps=timestamps,
        diarization=diarization,
        engine=engine,
        transport=transport,
        base_url=base_url,
    )
    if engine is not None:
        _record_cost(engine, provider_name, model_name, result.duration_seconds)
    return result


def _is_transcription_model(provider: str, model: str, engine: Any) -> bool:
    """Whether this model is served by a transcription endpoint.

    From the catalog's own `type`, not from a name pattern: `whisper-1` is
    obvious and `gpt-4o-transcribe` is not, and a pattern would have to be
    updated every time a provider names one differently.
    """
    from .catalog.catalog import resolve_catalog

    if provider != "openai":
        return False
    catalog = resolve_catalog(engine.catalog if engine is not None else None)
    info = catalog.get(provider, model)
    return info is not None and dict(info).get("type") == "stt"


def _structured_asked_for(
    languages: Sequence[str] | None,
    keywords: Sequence[str] | None,
    timestamps: str | None,
    diarization: bool,
) -> list[str]:
    asked = []
    if languages:
        asked.append("languages")
    if keywords:
        asked.append("keywords")
    if timestamps:
        asked.append("timestamps")
    if diarization:
        asked.append("diarization")
    return asked


def _openai_transcribe(
    *,
    data: bytes,
    mime_type: str,
    model: str,
    api_key: str,
    language: str | None,
    languages: Sequence[str] | None,
    keywords: Sequence[str] | None,
    timestamps: str | None,
    diarization: bool,
    engine: Any,
    transport: Any,
    base_url: str | None,
) -> TranscriptionResult:
    from .llm.wire_multipart import MultipartFile, encode_multipart, to_form_data
    from .llm.wire_transforms import make_registry
    from .wire.interpreter import build_from_spec
    from .wire.registry import get_wire_spec

    catalog_model = model
    if engine is not None:
        catalog_model = engine.catalog.resolve_model_id("openai", model)

    built = build_from_spec(
        get_wire_spec("openai/transcriptions"),
        {
            "model": catalog_model,
            "language": language,
            "languages": list(languages or ()),
            "keywords": list(keywords or ()),
            "wordTimestamps": bool(timestamps),
            "diarization": diarization,
        },
        make_registry({}),
        "openai",
        None,
        {"apiKey": api_key, "baseURL": base_url or _BASE_URL},
    )

    form = to_form_data(
        built.multipart or [],
        MultipartFile(filename=_filename_for(mime_type), data=data, mime_type=mime_type),
    )
    # Encoded to bytes with `rawBody`, which is how the engine is told not to
    # re-serialise a body that already is one. Passing the parts through under a
    # key of their own reached nothing: the request went out with an empty body
    # and OpenAI answered "you must provide a model parameter".
    encoded, content_type = encode_multipart(form)
    request: dict[str, Any] = {
        "url": built.url,
        "method": built.method or "POST",
        "headers": {**dict(built.headers or {}), "content-type": content_type},
        "body": encoded,
        "rawBody": True,
        "provider": "openai",
        "model": catalog_model,
        "responseType": "json",
    }

    response = _fetch(engine, transport)(request)
    status = response.get("status") if isinstance(response, Mapping) else 0
    body = (response.get("body") if isinstance(response, Mapping) else {}) or {}
    if not isinstance(status, int) or status >= 400:
        raise RuntimeError(f"transcription failed ({status}): {body}")

    parsed = parse_transcription(body)
    from dataclasses import replace

    return replace(parsed, model=model)


def _via_completion(
    *,
    model: str,
    provider: str,
    audio: Any,
    api_key: str,
    prompt: str | None,
    engine: Any,
    transport: Any,
    requested: Sequence[str],
) -> TranscriptionResult:
    """On a generateContent provider, STT is a normal completion.

    Google types an audio-transcription config and even validates it, then
    returns a response byte-identical to one sent without it: no speaker labels,
    no word timings. Rather than emit a field proven inert, this WARNS -- a
    caller who asked for speakers must learn they are not coming.
    """
    from .helpers.one_shot import complete

    if requested and engine is not None:
        engine.hooks.emit_sync(
            "onWarning",
            {
                "source": "media",
                "code": "transcription_option_unsupported",
                "message": (
                    f"transcribe: {', '.join(requested)} "
                    f"{'is' if len(requested) == 1 else 'are'} not available on "
                    f"{provider!r} -- the transcript will be plain text. Structured "
                    f"transcription currently requires an OpenAI transcription model."
                ),
                "details": {"provider": provider, "requested": list(requested)},
            },
        )

    options: dict[str, Any] = {"api_key": api_key, "max_tokens": 1024}
    if engine is not None:
        options["engine"] = engine
    if transport is not None:
        options["transport"] = transport

    answer = complete(
        model=model,
        prompt=prompt or DEFAULT_TRANSCRIBE_PROMPT,
        attachments=[audio],
        **options,
    )
    return TranscriptionResult(text=answer.text, model=model)


def _fetch(engine: Any, transport: Any) -> Any:
    from .bus.hook_bus import HookBus
    from .network.executor import RequestExecutor
    from .network.retry import DEFAULT_RETRY
    from .transport import as_fetch, http_transport

    if engine is not None and transport is None:
        return engine.fetch

    send = as_fetch(transport or http_transport())
    executor = RequestExecutor(engine.hooks if engine is not None else HookBus())

    def fetch(req: Any, options: Any = None) -> Any:
        return executor.execute(req, lambda r: send(r), DEFAULT_RETRY)

    return fetch


def _record_cost(engine: Any, provider: str, model: str, seconds: float | None) -> None:
    """Transcription is billed by DURATION, not by tokens.

    Recorded with an explicit zero token count rather than left out: a run whose
    ledger simply omits its transcriptions reads as one that made none.
    """
    from .cost import Cost
    from .cost_collector import CostEntry

    engine.hooks.emit_sync(
        "onCostEntry",
        CostEntry(
            id=f"cost_{uuid.uuid4().hex[:12]}",
            timestamp=time.time() * 1000,
            provider=provider,
            model=model,
            tokens={"input": 0, "output": 0, "cached": 0, "cache_write": 0, "reasoning": 0},
            cost=None if seconds is None else Cost(total=0.0, source="calculated"),
            tags={
                "provider": provider,
                "model": model,
                "type": "transcription",
                "seconds": seconds,
            },
        ),
    )


__all__ = [
    "DEFAULT_TRANSCRIBE_PROMPT",
    "STRUCTURED_ONLY",
    "TranscriptSegment",
    "TranscriptWord",
    "TranscriptionResult",
    "parse_transcription",
    "transcribe",
]
