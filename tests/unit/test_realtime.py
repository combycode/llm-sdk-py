"""Live sessions: the frames that go out, and the events that come back.

The outbound half is a differential against `tests/fixtures/service-golden.json`
-- fifteen artifacts frozen from the TypeScript adapters, covering both
providers in both modalities. A realtime session has no request/response pair to
compare, so what is frozen is the three things it actually produces: the
connection descriptor, the handshake frame, and the frames of each turn.

The inbound half runs whole sessions against a fake socket. That is not a
weaker test than a live one for this code: the interesting behaviour is
readiness, buffering and threading, and a real provider exercises exactly one
path through it while a fake can be made to close early, answer late, or send a
frame that is not JSON.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.events import (
    AudioEvent,
    RealtimeErrorEvent,
    SessionOpenEvent,
    TextEvent,
)
from combycode_llm_sdk.helpers.realtime import Realtime
from combycode_llm_sdk.realtime import (
    GoogleRealtimeAdapter,
    OpenAIRealtimeAdapter,
    RealtimeConnection,
    SessionConfig,
    Turn,
    frame_of,
    google_usage,
    modalities_of,
    openai_usage,
)

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "service-golden.json").read_text("utf-8")
)["index"]

#: The four bytes the freeze script used as audio.
PCM = bytes([0, 1, 2, 3])

#: Exactly the corpus the TypeScript froze: two providers, two modalities each.
CASES: list[dict[str, Any]] = [
    {
        "key": "openai/text",
        "provider": "openai",
        "config": SessionConfig(model="gpt-realtime"),
        "turns": [
            ("text", Turn(text="hello"), True),
            # turnComplete=False withholds response.create and leaves the turn
            # open -- where Gemini expresses the same thing as a field.
            ("text.open", Turn(text="hello"), False),
        ],
    },
    {
        "key": "openai/audio",
        "provider": "openai",
        "config": SessionConfig(
            model="gpt-realtime",
            modalities=("text", "audio"),
            voice="alloy",
            instructions="be brief",
        ),
        "turns": [
            ("audio", Turn(audio=PCM), True),
            ("both", Turn(text="and this", audio=PCM), True),
        ],
    },
    {
        "key": "google/text",
        "provider": "google",
        "config": SessionConfig(model="gemini-3.1-flash-live-preview"),
        "turns": [
            ("text", Turn(text="hello"), True),
            ("text.open", Turn(text="hello"), False),
        ],
    },
    {
        "key": "google/audio",
        "provider": "google",
        "config": SessionConfig(
            model="models/gemini-3.1-flash-live-preview",
            modalities=("audio",),
            voice="Kore",
            instructions="be brief",
        ),
        "turns": [("audio", Turn(audio=PCM), True)],
    },
]

ADAPTERS: dict[str, Any] = {
    "openai": OpenAIRealtimeAdapter,
    "google": GoogleRealtimeAdapter,
}


def canon(value: Any) -> Any:
    """Key-sorted, `None` dropped -- the same shape the freeze recorded."""
    if isinstance(value, list):
        return [canon(v) for v in value]
    if isinstance(value, dict):
        return {k: canon(value[k]) for k in sorted(value) if value[k] is not None}
    return value


def connect_request_wire(adapter: Any, config: SessionConfig) -> Any:
    request = adapter.build_connect_request(config)
    wire: dict[str, Any] = {
        "url": request.url,
        "provider": request.provider,
        "model": request.model,
    }
    if request.protocols:
        wire["protocols"] = list(request.protocols)
    return canon(wire)


class TestTheSameFramesAsTheTypeScript:
    """Fifteen frozen artifacts, held to what the other implementation built."""

    @pytest.mark.parametrize("case", CASES, ids=[c["key"] for c in CASES])
    def test_the_connection_descriptor_matches(self, case: dict[str, Any]) -> None:
        adapter = ADAPTERS[case["provider"]](api_key="k")
        wire = connect_request_wire(adapter, case["config"])
        assert wire == canon(GOLDEN[f"realtime/{case['key']}/connect"])

    @pytest.mark.parametrize("case", CASES, ids=[c["key"] for c in CASES])
    def test_the_handshake_frame_matches(self, case: dict[str, Any]) -> None:
        adapter = ADAPTERS[case["provider"]](api_key="k")
        frame = adapter.build_open_frame(case["config"])
        assert canon(frame) == canon(GOLDEN[f"realtime/{case['key']}/open"])

    @pytest.mark.parametrize(
        ("case", "turn"),
        [(c, t) for c in CASES for t in c["turns"]],
        ids=[f"{c['key']}/{t[0]}" for c in CASES for t in c["turns"]],
    )
    def test_the_turn_frames_match(self, case: dict[str, Any], turn: Any) -> None:
        name, payload, complete = turn
        adapter = ADAPTERS[case["provider"]](api_key="k")
        frames = adapter.build_turn_frames(payload, complete)
        assert canon(frames) == canon(GOLDEN[f"realtime/{case['key']}/turn.{name}"])

    def test_the_golden_covers_every_case_and_no_more(self) -> None:
        # A differential that quietly stops covering a case keeps reporting
        # green over an adapter nobody measures any more.
        frozen = {k for k in GOLDEN if k.startswith("realtime/")}
        expected = set()
        for case in CASES:
            expected.add(f"realtime/{case['key']}/connect")
            expected.add(f"realtime/{case['key']}/open")
            for name, _payload, _complete in case["turns"]:
                expected.add(f"realtime/{case['key']}/turn.{name}")
        assert frozen == expected

    def test_the_comparison_can_fail(self) -> None:
        # The canary: every assertion above passes trivially if the comparison
        # cannot discriminate, so one known-wrong build must be rejected.
        adapter = OpenAIRealtimeAdapter(api_key="wrong-key")
        wire = connect_request_wire(adapter, SessionConfig(model="gpt-realtime"))
        assert wire != canon(GOLDEN["realtime/openai/text/connect"])


class TestWhatEachProviderNeedsThatTheOtherDoesNot:
    """The four asymmetries, asserted rather than described in a comment."""

    def test_openai_carries_its_key_in_a_subprotocol(self) -> None:
        # Not a header and not a query parameter: a browser cannot set a header
        # on a WebSocket handshake, which is why `protocols` is modelled at all.
        request = OpenAIRealtimeAdapter(api_key="sk-secret").build_connect_request(
            SessionConfig(model="gpt-realtime")
        )
        assert request.protocols == ("realtime", "openai-insecure-api-key.sk-secret")
        assert "sk-secret" not in request.url

    def test_google_carries_its_key_in_the_query_string(self) -> None:
        # The only place in this library where a key rides in a URL, and for the
        # same reason: it is a WebSocket handshake.
        request = GoogleRealtimeAdapter(api_key="gk").build_connect_request(
            SessionConfig(model="gemini-live")
        )
        assert request.url.endswith("?key=gk")
        assert request.protocols is None

    def test_openai_names_the_model_in_the_url_and_google_in_the_frame(self) -> None:
        openai = OpenAIRealtimeAdapter(api_key="k")
        google = GoogleRealtimeAdapter(api_key="k")
        config = SessionConfig(model="zephyr-live")

        assert "model=zephyr-live" in openai.build_connect_request(config).url
        assert "zephyr-live" not in google.build_connect_request(config).url
        assert google.build_open_frame(config)["setup"]["model"] == "models/zephyr-live"

    def test_only_openai_needs_a_second_frame_to_ask_for_a_reply(self) -> None:
        turn = Turn(text="hi")
        openai = OpenAIRealtimeAdapter(api_key="k").build_turn_frames(turn, True)
        google = GoogleRealtimeAdapter(api_key="k").build_turn_frames(turn, True)

        assert [f["type"] for f in openai] == [
            "conversation.item.create",
            "response.create",
        ]
        assert len(google) == 1
        assert google[0]["clientContent"]["turnComplete"] is True

    def test_a_google_base_url_is_coerced_to_a_websocket_scheme(self) -> None:
        # A caller pointing at a proxy writes an http:// URL, because that is
        # what every other base URL in this library is.
        adapter = GoogleRealtimeAdapter(api_key="k", base_url="https://proxy.example/")
        assert adapter.build_connect_request(SessionConfig(model="m")).url.startswith(
            "wss://proxy.example/ws/"
        )


# ── a socket that answers ───────────────────────────────────────────────────


class FakeSocket:
    """A scriptable WebSocket. Answers a send with whatever it was told to."""

    def __init__(self, script: Any = None, binary: bool = False) -> None:
        self.sent: list[Any] = []
        self.closed = False
        self._binary = binary
        self._script: Any = script or (lambda message: [])
        self._inbox: list[Any] = []
        self._lock = threading.Lock()

    def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))
        replies = self._script(json.loads(data))
        if replies:
            with self._lock:
                self._inbox.extend(replies)

    def send_bytes(self, data: bytes) -> None:  # pragma: no cover -- unused here
        self.sent.append(data)

    def push(self, *messages: Any) -> None:
        with self._lock:
            self._inbox.extend(messages)

    def receive(self, timeout: float | None = None) -> Any:
        with self._lock:
            if self._inbox:
                item = self._inbox.pop(0)
        if "item" not in dir():
            raise TimeoutError
        if isinstance(item, BaseException):
            raise item
        payload = json.dumps(item)
        data: Any = payload.encode("utf-8") if self._binary else payload
        return type("Event", (), {"data": data})()

    def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = True


def socket_factory(socket: FakeSocket) -> Any:
    class Held:
        def __enter__(self) -> FakeSocket:
            return socket

        def __exit__(self, *exc: object) -> None:
            return None

    def connect(url: str, **kwargs: Any) -> Any:
        socket.url = url  # type: ignore[attr-defined]
        socket.kwargs = kwargs  # type: ignore[attr-defined]
        return Held()

    return connect


def openai_script(message: dict[str, Any]) -> list[dict[str, Any]]:
    if message.get("type") != "response.create":
        return []
    return [
        {"type": "response.output_text.delta", "delta": "Par"},
        {"type": "response.output_text.delta", "delta": "is"},
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                    "input_token_details": {"audio_tokens": 4, "cached_tokens": 2},
                    "output_token_details": {"audio_tokens": 1},
                }
            },
        },
    ]


#: A session that never becomes ready would hang `for event in session` forever,
#: and a hung suite says nothing about what broke. Every read below is bounded,
#: so a regression in readiness or framing FAILS, naming what did not arrive.
READ_TIMEOUT_SECONDS = 5.0


def read_until(session: Any, kind: str, timeout: float = READ_TIMEOUT_SECONDS) -> list[Any]:
    """Events up to and including the first of `kind`, or an assertion failure."""
    seen: list[Any] = []
    done = threading.Event()

    def drain() -> None:
        for event in session:
            seen.append(event)
            if event.type == kind:
                break
        done.set()

    threading.Thread(target=drain, daemon=True).start()
    if not done.wait(timeout):
        raise AssertionError(
            f"no {kind!r} event within {timeout:g}s; saw {[e.type for e in seen]}"
        )
    return seen


class TestAWholeSession:
    def session(self, socket: FakeSocket, **over: Any) -> Realtime:
        options: dict[str, Any] = {
            "model": "openai/gpt-realtime",
            "api_key": "k",
            "connect": socket_factory(socket),
        }
        options.update(over)
        return Realtime(**options)

    def test_a_turn_arrives_as_events_in_order(self) -> None:
        socket = FakeSocket(openai_script)
        with self.session(socket) as session:
            session.send("Name the capital of France.")
            seen = read_until(session, "turn_end")
            kinds = [e.type for e in seen]
            text = "".join(e.text for e in seen if isinstance(e, TextEvent))

        assert text == "Paris"
        assert kinds == ["open", "text", "text", "usage", "turn_end"]

    def test_the_handshake_goes_out_before_the_turn(self) -> None:
        socket = FakeSocket(openai_script)
        with self.session(socket, instructions="Be brief.") as session:
            session.send("hi")
            read_until(session, "turn_end")

        assert [f["type"] for f in socket.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]
        assert socket.sent[0]["session"]["instructions"] == "Be brief."

    def test_a_send_before_the_socket_opens_is_buffered_not_lost(self) -> None:
        # Gemini is not ready until it answers `setupComplete`, so a caller
        # writing open-send-read would otherwise drop their first turn.
        socket = FakeSocket(binary=True)
        session = Realtime(
            model="google/gemini-live", api_key="k", modalities=["audio"],
            connect=socket_factory(socket),
        )
        with session:
            session.send("hello")
            # Only the setup frame so far: the turn is held.
            assert [next(iter(f)) for f in socket.sent] == ["setup"]
            socket.push({"setupComplete": {}})
            opened = read_until(session, "open")
            assert isinstance(opened[-1], SessionOpenEvent)

        assert [next(iter(f)) for f in socket.sent] == ["setup", "clientContent"]

    def test_google_audio_arrives_decoded(self) -> None:
        socket = FakeSocket(binary=True)
        session = Realtime(
            model="google/gemini-live", api_key="k", modalities=["audio"],
            connect=socket_factory(socket),
        )
        with session:
            socket.push(
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/pcm;rate=24000",
                                        "data": "AAECAw==",
                                    }
                                }
                            ]
                        },
                        "turnComplete": True,
                    }
                },
            )
            seen = read_until(session, "turn_end")

        chunks = [e for e in seen if isinstance(e, AudioEvent)]
        assert b"".join(c.audio for c in chunks) == PCM
        # Reported as the provider sent it: a caller writing a WAV header needs
        # the rate parameter.
        assert [c.mime_type for c in chunks] == ["audio/pcm;rate=24000"]
        assert [c.sample_rate for c in chunks] == [24_000]

    def test_a_provider_error_is_an_event_carrying_its_message(self) -> None:
        socket = FakeSocket()
        with self.session(socket) as session:
            socket.push({"type": "error", "error": {"message": "session expired"}})
            failed = read_until(session, "error")[-1]

        assert isinstance(failed, RealtimeErrorEvent)
        assert failed.message == "session expired"

    def test_a_frame_that_is_not_json_is_dropped_rather_than_fatal(self) -> None:
        socket = FakeSocket()
        with self.session(socket) as session:
            # A keepalive, a banner, anything. The session must survive it.
            socket.push(
                "not json at all",
                {"type": "response.output_text.delta", "delta": "still here"},
            )
            survived = read_until(session, "text")[-1]

        assert isinstance(survived, TextEvent)
        assert survived.text == "still here"

    def test_the_iteration_ends_when_the_socket_closes(self) -> None:
        # There is no `close` event to miss: ending the iteration is how Python
        # already says it, so a caller's `for` loop simply finishes.
        socket = FakeSocket()
        session = self.session(socket).open()
        seen: list[Any] = []

        def drain() -> None:
            seen.extend(session)

        reader = threading.Thread(target=drain)
        reader.start()
        session.close()
        reader.join(timeout=5)

        assert not reader.is_alive(), "the iterator did not stop when the socket closed"
        assert socket.closed is False, "the context manager owns the close, not us"

    def test_sending_before_open_says_so(self) -> None:
        socket = FakeSocket()
        session = self.session(socket)
        with pytest.raises(RuntimeError, match="not open"):
            session.send("hi")

    def test_send_with_nothing_in_it_is_refused(self) -> None:
        socket = FakeSocket(openai_script)
        with (
            self.session(socket) as session,
            pytest.raises(ValueError, match="text, audio, or both"),
        ):
            session.send()


class TestTheEngineSeesIt:
    def test_usage_reaches_the_cost_pipeline(self) -> None:
        # A live session that never emitted onCompletion would be invisible in
        # `engine.cost` and first appear on the invoice.
        from combycode_llm_sdk.helpers.engine import Engine

        engine = Engine(api_keys={"openai": "k"}, register_as_default=False)
        completions: list[Any] = []
        engine.hooks.on("onCompletion", lambda ctx: completions.append(dict(ctx)))

        socket = FakeSocket(openai_script)
        with Realtime(
            model="openai/gpt-realtime", engine=engine, connect=socket_factory(socket)
        ) as session:
            session.send("hi")
            read_until(session, "turn_end")

        assert len(completions) == 1
        usage = completions[0]["response"]["usage"]
        assert usage["totalTokens"] == 13
        assert usage["audioInputTokens"] == 4

    def test_the_frame_hooks_fire_in_both_directions(self) -> None:
        hooks = HookBus()
        frames: list[dict[str, Any]] = []
        opened: list[dict[str, Any]] = []
        hooks.on("onRealtimeFrame", lambda ctx: frames.append(dict(ctx)))
        hooks.on("onRealtimeOpen", lambda ctx: opened.append(dict(ctx)))

        socket = FakeSocket(openai_script)
        adapter = OpenAIRealtimeAdapter(api_key="k")
        request = adapter.build_connect_request(SessionConfig(model="gpt-realtime"))
        connection = RealtimeConnection(request, hooks, connect=socket_factory(socket))
        session = adapter.connect(SessionConfig(model="gpt-realtime"), lambda _r: connection)
        connection.open()
        session.send("hi")
        read_until(session, "turn_end")
        connection.close()

        assert [o["url"] for o in opened] == [request.url]
        assert {f["direction"] for f in frames} == {"in", "out"}
        assert all(f["provider"] == "openai" for f in frames)
        assert all(f["bytes"] > 0 for f in frames)


class TestReadingTheProvidersNumbers:
    def test_openai_splits_audio_from_text_tokens(self) -> None:
        # They price differently, so billing a spoken answer at the text rate is
        # not a rounding error -- it is the wrong number.
        usage = openai_usage(
            {
                "input_tokens": 10,
                "output_tokens": 6,
                "total_tokens": 16,
                "input_token_details": {"audio_tokens": 4, "cached_tokens": 2},
                "output_token_details": {"audio_tokens": 5},
            }
        )
        assert (usage.input_tokens, usage.audio_input_tokens) == (6, 4)
        assert (usage.output_tokens, usage.audio_output_tokens) == (1, 5)
        assert usage.cached_tokens == 2

    def test_a_text_only_turn_reports_no_audio_rather_than_zero(self) -> None:
        # `Usage` treats 0 as "billed for silence", which prices a text turn at
        # audio rates.
        usage = openai_usage({"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
        assert usage.audio_input_tokens is None
        assert usage.audio_output_tokens is None
        assert (usage.input_tokens, usage.output_tokens) == (3, 2)

    def test_google_reads_either_name_for_the_output_count(self) -> None:
        assert google_usage({"responseTokenCount": 7}).output_tokens == 7
        assert google_usage({"candidatesTokenCount": 9}).output_tokens == 9


class TestTheSmallPieces:
    @pytest.mark.parametrize(
        ("given", "wanted"),
        [(None, ("text",)), ([], ("text",)), (["audio"], ("audio",)),
         (["text", "audio"], ("text", "audio")), (["AUDIO"], ("audio",))],
    )
    def test_modalities_are_normalised(self, given: Any, wanted: Any) -> None:
        assert modalities_of(given) == wanted

    def test_an_unknown_modality_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="video"):
            modalities_of(["video"])

    def test_a_frame_is_recognised_by_its_payload_not_its_class(self) -> None:
        # Duck-typed so `wsproto` is never imported and a caller's own client
        # only has to produce something with a `.data`.
        assert frame_of(type("E", (), {"data": "hi"})()) == __import__(
            "combycode_llm_sdk.realtime", fromlist=["Frame"]
        ).Frame(text="hi")
        binary = frame_of(type("E", (), {"data": b"hi"})())
        assert binary is not None and binary.binary == b"hi"
        # A ping is not a frame.
        assert frame_of(type("E", (), {"data": None})()) is None
        assert frame_of(object()) is None

    def test_a_provider_with_no_socket_api_says_so(self) -> None:
        with pytest.raises(ValueError, match="anthropic"):
            Realtime(model="anthropic/claude-haiku-4.5", api_key="k")

    def test_a_missing_key_says_which_provider(self) -> None:
        from combycode_llm_sdk.helpers.engine import Engine

        with pytest.raises(ValueError, match="openai"):
            Realtime(
                model="openai/gpt-realtime",
                engine=Engine(register_as_default=False),
            )
