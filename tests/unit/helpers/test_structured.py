"""`structured=SomeDataclass` -- scenarios 11 and 12.

The bug these exist for: `complete(structured=Weather)` sent the CLASS to the
wire, which became an empty schema, and Anthropic answered "Empty schema ({})
that accepts any JSON". Google and xAI accepted it and returned prose, and the
runner still printed PASS because the scenario returns `""` on a failed parse --
so the corpus reported green on three of five cells that were all broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from combycode_llm_sdk import complete
from combycode_llm_sdk.helpers.json_schema import SchemaError
from combycode_llm_sdk.helpers.structured import instantiate, parse_into, to_wire
from combycode_llm_sdk.transport import TransportRequest, TransportResponse


@dataclass
class Weather:
    city: str
    temp_c: float


@dataclass
class Address:
    city: str
    country: str


@dataclass
class Person:
    name: str
    age: int
    address: Address
    nickname: str | None = None


@dataclass
class Team:
    name: str
    members: list[Person] = field(default_factory=list)


class Stub:
    def __init__(self, text: str) -> None:
        self.text = text
        self.seen: list[TransportRequest] = []

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.seen.append(request)
        return TransportResponse(
            status=200,
            body={
                "id": "msg_1",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": self.text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )


def wire(cls: Any) -> dict[str, Any]:
    """`to_wire` and the assertion that it produced something.

    `to_wire` returns None only for `structured=None`, which no test here passes.
    """
    sent = to_wire(cls)
    assert sent is not None
    return sent


def one(stub: Stub, **over: Any) -> Any:
    kwargs: dict[str, Any] = {
        "model": "anthropic/claude-haiku-4.5",
        "api_key": "k",
        "transport": stub,
        "prompt": "extract it",
        "max_tokens": 64,
    }
    kwargs.update(over)
    return complete(**kwargs)


class TestTheSchemaOnTheWire:
    def test_a_dataclass_reaches_the_wire_as_its_schema(self) -> None:
        # The regression. The class itself became `{}`, and the provider said so.
        stub = Stub('{"city": "Paris", "temp_c": 20}')
        one(stub, structured=Weather)
        sent = stub.seen[0].body["output_config"]["format"]["schema"]
        assert sent["properties"]["city"] == {"type": "string"}
        assert sent["properties"]["temp_c"] == {"type": "number"}
        assert sent != {}

    def test_every_field_is_required_and_optionals_are_nullable(self) -> None:
        # OpenAI's strict mode refuses a schema whose properties are not all
        # required, and the wire layer turns strict OFF for a schema that cannot
        # satisfy it -- so an omitted optional silently downgrades the request.
        schema = wire(Person)["schema"]
        assert set(schema["required"]) == {"name", "age", "address", "nickname"}
        assert schema["properties"]["nickname"]["type"] == ["string", "null"]
        assert schema["additionalProperties"] is False

    def test_a_nested_dataclass_becomes_a_nested_object(self) -> None:
        schema = wire(Person)["schema"]
        assert schema["properties"]["address"]["properties"]["country"] == {"type": "string"}

    def test_a_list_of_dataclasses_becomes_an_array_of_objects(self) -> None:
        schema = wire(Team)["schema"]
        items = schema["properties"]["members"]
        assert items["type"] == ["array", "null"]
        assert items["items"]["properties"]["name"] == {"type": "string"}

    def test_the_class_name_travels_with_the_schema(self) -> None:
        assert wire(Weather)["name"] == "Weather"

    def test_a_raw_schema_mapping_still_works(self) -> None:
        given = {"schema": {"type": "object", "properties": {}}}
        assert to_wire(given) == given

    def test_an_instance_is_refused_with_the_fix_named(self) -> None:
        with pytest.raises(SchemaError, match="not an instance"):
            to_wire(Weather(city="Paris", temp_c=20.0))

    def test_nothing_asked_for_sends_nothing(self) -> None:
        assert to_wire(None) is None


class TestTheAnswerComesBack:
    def test_parsed_is_a_real_instance(self) -> None:
        stub = Stub('{"city": "Paris", "temp_c": 20}')
        got = one(stub, structured=Weather)
        assert isinstance(got.parsed, Weather)
        assert got.parsed.city == "Paris"

    def test_an_integer_for_a_float_field_is_widened(self) -> None:
        # JSON has one number type, so `20` for a float is right and only its
        # spelling is not.
        got = parse_into(Weather, '{"city": "Paris", "temp_c": 20}')
        assert got.temp_c == 20.0
        assert isinstance(got.temp_c, float)

    def test_a_float_for_an_int_field_is_left_alone(self) -> None:
        # Truncating 20.7 to 20 would hide a real disagreement.
        got = instantiate(Person, {"name": "A", "age": 36.7, "address": {"city": "L", "country": "UK"}})
        assert got.age == 36.7

    def test_nested_objects_become_nested_instances(self) -> None:
        got = parse_into(
            Person,
            '{"name": "Ada", "age": 36, "address": {"city": "London", '
            '"country": "United Kingdom"}, "nickname": null}',
        )
        assert isinstance(got.address, Address)
        assert got.address.city == "London"
        assert got.nickname is None

    def test_a_list_of_objects_becomes_a_list_of_instances(self) -> None:
        got = parse_into(
            Team,
            '{"name": "T", "members": [{"name": "Ada", "age": 36, '
            '"address": {"city": "L", "country": "UK"}}]}',
        )
        assert isinstance(got.members[0], Person)
        assert got.members[0].address.country == "UK"

    def test_a_fenced_answer_is_unwrapped(self) -> None:
        stub = Stub('```json\n{"city": "Paris", "temp_c": 20}\n```')
        assert one(stub, structured=Weather).parsed.city == "Paris"

    def test_an_omitted_optional_takes_its_default(self) -> None:
        got = parse_into(
            Person, '{"name": "Ada", "age": 36, "address": {"city": "L", "country": "UK"}}'
        )
        assert got.nickname is None

    def test_an_explicit_null_leaves_a_default_factory_alone(self) -> None:
        # `members` is annotated `list`, never `list | None`, so writing None into
        # it would put a value there the annotation does not allow.
        got = parse_into(Team, '{"name": "T", "members": null}')
        assert got.members == []

    def test_an_unknown_key_is_dropped_rather_than_fatal(self) -> None:
        # Every field the caller asked for is present; discarding the answer over
        # one stray key helps nobody.
        got = parse_into(Weather, '{"city": "Paris", "temp_c": 20, "humidity": 60}')
        assert got.city == "Paris"

    def test_a_raw_schema_gets_the_decoded_json_not_an_instance(self) -> None:
        stub = Stub('{"n": 7}')
        got = one(stub, structured={"schema": {"type": "object"}})
        assert got.parsed == {"n": 7}


class TestWhenTheAnswerDisappoints:
    def test_unparseable_leaves_parsed_none_and_text_intact(self) -> None:
        # `12` requires exactly this: raising would make a disappointing answer
        # indistinguishable from a broken request, and a retry awkward to write.
        stub = Stub("I am afraid I cannot do that.")
        got = one(stub, structured=Weather)
        assert got.parsed is None
        assert got.text == "I am afraid I cannot do that."

    def test_json_of_the_wrong_shape_also_leaves_parsed_none(self) -> None:
        stub = Stub('{"something": "else"}')
        got = one(stub, structured=Weather)
        assert got.parsed is None
        assert got.text == '{"something": "else"}'

    def test_a_missing_required_field_is_named(self) -> None:
        with pytest.raises(SchemaError, match="missing temp_c"):
            parse_into(Weather, '{"city": "Paris"}')

    def test_an_array_where_an_object_was_asked_for_is_named(self) -> None:
        with pytest.raises(SchemaError, match="expected an object"):
            parse_into(Weather, "[1, 2, 3]")
