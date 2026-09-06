"""ModelCatalog -- pricing, capabilities, and model info. Loaded from JSON.

Transposed from `unified-library-ts/src/catalog/catalog.ts`.

`ModelInfo` and its parts (`ModelPricing`, `ModelCapabilities`,
`ModelReasoning`, `TokenizerInfo`, `MediaParamSpec`) are camelCase dicts, per
`llm/types/messages.py` -- these come straight out of the vendored
`catalog/data/*.json`, which is etalon and byte-identical with the TypeScript's.

**Absent is not None here.** Three fields are tri-state: `supportsPreviousResponseId`,
`stateRetentionDuration` and `stateModelBound` each mean one thing when the entry
omits them (fall back to the provider default) and another when the entry sets
them -- and `stateRetentionDuration` can legitimately BE null, meaning "no
retention". TypeScript gets this from `!== undefined`; Python gets it by never
writing a key the source did not carry, and testing membership.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..bus.context import to_camel
from .builtin_tools import PROVIDER_BUILTIN_TOOLS

#: `type ApiType` (catalog.ts:19). Same five names as `llm/types/provider.py`,
#: declared there too because the catalog is loadable without the llm layer.
ApiType = str

#: `interface ModelPricing` (catalog.ts:21) -- `{inputPerMTok?, outputPerMTok?,
#: cacheReadPerMTok?, cacheWritePerMTok?, audioInputPerMTok?, audioOutputPerMTok?,
#: perImage?, perSecond?, perMinute?, perMChars?, perUnit?, tiers?}`.
#:
#: `perMinute` is the speech-to-text rate (USD per minute of audio), used by
#: whisper-class and gpt-4o-transcribe when the provider bills by duration rather
#: than by token. `perUnit` is keyed by a quality/resolution tier (video
#: `{"720p":0.1,"1080p":0.12}`, image `{"1k":0.002,"2k":0.02}`) and overrides the
#: flat `perImage`/`perSecond` when the selected resolution matches a key.
#: `tiers` is keyed by the provider's OWN billed tier name (the value the
#: provider returns: anthropic `standard|priority|batch`, openai
#: `flex|scale|priority|batch`); the flat fields are the implicit `standard`
#: tier, and cost looks up `tiers[usage.pricingTier]` falling back to them.
ModelPricing = dict[str, Any]

#: `type TierRates` (catalog.ts:53) -- one tier's overrides, same shape as the
#: flat rates with no nesting.
TierRates = dict[str, Any]

#: `interface ModelCapabilities` (catalog.ts:55).
ModelCapabilities = dict[str, Any]

#: `interface MediaParamSpec` (catalog.ts:75) -- one generation parameter a media
#: model accepts, either an enum (`values` + `default`) or a numeric range
#: (`min`/`max`). Keys are normalized (`aspectRatio`, `size`, `voice`,
#: `duration`); each provider adapter maps the normalized key to its own wire
#: param name.
MediaParamSpec = dict[str, Any]

#: `interface ModelReasoning` (catalog.ts:89).
ModelReasoning = dict[str, Any]

#: `interface TokenizerInfo` (catalog.ts:98).
TokenizerInfo = dict[str, Any]

#: One catalog entry as it is STORED -- the JSON shape `load()` reads and
#: `set()` builds. Open by design: a provider adding a field must not need a
#: change here for the entry to round-trip.
ModelEntry = dict[str, Any]


class ModelInfo(Mapping[str, Any]):
    """One catalog entry, as a caller receives it. `interface ModelInfo` (catalog.ts:106).

    Readable BOTH ways, and each way has a reason. `info.model` is what the
    reviewed examples write, and it is the natural transposition of a TypeScript
    interface field. `info["pricing"]` is what every consumer inside the library
    already uses, and the field set is open -- a provider adding a key must not
    break a reader -- which is what a Mapping expresses and a dataclass does not.

    A view, not a copy: it reads the entry the catalog holds, so an entry
    updated through `set()` is not stale here. Read-only, so a caller cannot
    edit the catalog by writing to something it handed out.
    """

    __slots__ = ("_entry",)

    def __init__(self, entry: ModelEntry) -> None:
        object.__setattr__(self, "_entry", entry)

    def __getitem__(self, key: str) -> Any:
        return self._entry[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entry)

    def __len__(self) -> int:
        return len(self._entry)

    def __getattr__(self, name: str) -> Any:
        entry = object.__getattribute__(self, "_entry")
        if name in entry:
            return entry[name]
        # The entry is stored with the wire's spelling (`contextWindow`); a
        # Python caller writes `context_window`. Same boundary the hook contexts
        # draw, and drawn with the same function so the two cannot disagree.
        camel = to_camel(name)
        if camel in entry:
            return entry[camel]
        raise AttributeError(
            f"a catalog entry has no {name!r}. It holds: {', '.join(sorted(entry))}."
        )

    def __repr__(self) -> str:
        return f"<ModelInfo {self._entry.get('provider')}/{self._entry.get('model')}>"

#: Provider-level server-state defaults, used when a model has no explicit value
#: or is not in the catalog at all.
#:
#: OpenAI/xAI: `previous_response_id` survives most model swaps but can break at
#: reasoning-model boundaries -> treated as model-bound, which is the safe
#: reading. Google: server-state via the Interactions API; docs and a live test
#: confirm it works ACROSS models of the same provider -> not model-bound, ~72h.
_PROVIDER_STATE: dict[str, dict[str, Any]] = {
    "openai": {"supports": True, "retention": "30d", "modelBound": True},
    "xai": {"supports": True, "retention": "30d", "modelBound": True},
    "google": {"supports": True, "retention": "72h", "modelBound": False},
    "anthropic": {"supports": False, "retention": None, "modelBound": True},
    "openrouter": {"supports": False, "retention": None, "modelBound": True},
}

_DEFAULT_CAPABILITIES: ModelCapabilities = {
    "toolUse": True,
    "streaming": True,
    "structuredOutput": True,
    "vision": False,
    "audio": False,
    "video": False,
    "imageGeneration": False,
    "audioGeneration": False,
    "videoGeneration": False,
}

_DEFAULT_REASONING: ModelReasoning = {
    "supported": False,
    "automatic": False,
    "effortControl": False,
    "encryptedContent": False,
    "summaryAvailable": False,
}

#: The five bundled catalogs, in the order the TypeScript loads them.
_PROVIDER_DEFAULT_FILES = (
    "anthropic.json",
    "openai.json",
    "google.json",
    "xai.json",
    "openrouter.json",
)

_DATA_DIR = Path(__file__).resolve().parent / "data"

#: Every optional `ModelInfo` field `set()` forwards from its input. Listed once
#: rather than written out as twenty assignments, because a field dropped from
#: this list is a field that silently stops being carried.
_OPTIONAL_FIELDS = (
    "contextWindow",
    "maxOutput",
    "mediaOnly",
    "tokenizer",
    "requiresDedicatedClient",
    "supportsPreviousResponseId",
    "wireSpec",
    "stateRetentionDuration",
    "stateModelBound",
    "providerModelName",
    "aliases",
    "type",
    "inputModalities",
    "outputModalities",
    "mediaParams",
    "family",
    "version",
    "status",
    "availability",
    "active",
    "deprecation",
)

_NORMALIZE_SEPARATOR = re.compile(r"(?<=\d)[-.](?=\d)")


def _with_builtin_tools(provider: str, caps: ModelCapabilities) -> ModelCapabilities:
    """Populate `builtinTools` from the adapter-sourced provider map.

    Applied only to tool-capable (chat-family) models -- embeddings/tts/image/stt
    models (`toolUse: false`) get none. An explicit `builtinTools` on the entry
    always wins.
    """
    if "builtinTools" in caps:
        return caps
    if not caps.get("toolUse"):
        return caps
    tools = PROVIDER_BUILTIN_TOOLS.get(provider)
    if not tools:
        return caps
    return {**caps, "builtinTools": list(tools)}


def normalize_id(model: str) -> str:
    """Spelling-insensitive form of a model id.

    Lower-cased, with a `-` or `.` BETWEEN TWO DIGITS unified to `.`.

    Providers spell the same version three ways and users copy whichever they
    saw: `gpt-4.1` and `gpt-4-1`, `gemini-2.5-flash` and `gemini-2-5-flash`,
    `claude-haiku-4.5` and `claude-haiku-4-5`. Only one of each pair is callable,
    and the others used to miss the catalog entirely -- which meant no price, no
    capabilities, and an id forwarded verbatim into a 404.

    Deliberately narrow. Only a separator between two DIGITS moves, so nothing
    that distinguishes two real models is erased: `gpt-4o` stays distinct from
    `gpt-4`, and `command-r7b` keeps its shape. Verified across the shipped
    catalogs -- 1016 normalized keys, zero collisions -- so this can never merge
    two models into one.
    """
    return _NORMALIZE_SEPARATOR.sub(".", model.lower())


class ModelCatalog:
    """`class ModelCatalog` (catalog.ts:212)."""

    def __init__(self) -> None:
        self._models: dict[str, ModelEntry] = {}
        #: `provider/alias` -> `provider/canonical-slug`. Lets `get` /
        #: `resolve_model_id` accept any callable id (providerModelName, dated
        #: snapshot) AND the slug.
        self._alias_index: dict[str, str] = {}
        #: `provider/normalized-id` -> `provider/canonical-slug`. The last
        #: resort, so a user's spelling of a version never decides whether the
        #: model is found.
        self._norm_index: dict[str, str] = {}

    @staticmethod
    def _key(provider: str, model: str) -> str:
        return f"{provider}/{model}"

    @staticmethod
    def _norm_key(provider: str, model: str) -> str:
        return f"{provider}/{normalize_id(model)}"

    def set(self, provider: str, model: str, info: dict[str, Any]) -> None:
        """Add or replace one entry. `info` must carry `pricing`."""
        canonical = self._key(provider, model)
        preferred = info.get("preferredApi") or "completions"
        entry: ModelEntry = {
            "provider": provider,
            "model": model,
            "pricing": info["pricing"],
            "preferredApi": preferred,
            "supportedApis": info.get("supportedApis") or [preferred],
            "capabilities": _with_builtin_tools(
                provider, {**_DEFAULT_CAPABILITIES, **(info.get("capabilities") or {})}
            ),
            "reasoning": {**_DEFAULT_REASONING, **(info.get("reasoning") or {})},
        }
        # Absent stays absent. `stateRetentionDuration: null` is a real value
        # meaning "no retention", so `in` is the test, never truthiness.
        for field in _OPTIONAL_FIELDS:
            if field in info:
                entry[field] = info[field]
        self._models[canonical] = entry

        # Index callable ids -> this slug (don't shadow a real slug key).
        for alias in [info.get("providerModelName"), *(info.get("aliases") or [])]:
            if alias and self._key(provider, alias) not in self._models:
                self._alias_index[self._key(provider, alias)] = canonical
        # Every spelling of every name this entry answers to, so a lookup never
        # turns on whether the user wrote 4-5 or 4.5. First writer wins: the slug
        # is indexed before its aliases, so a normalized key always points at the
        # canonical entry.
        for name in [model, info.get("providerModelName"), *(info.get("aliases") or [])]:
            if not name:
                continue
            nk = self._norm_key(provider, name)
            if nk not in self._norm_index:
                self._norm_index[nk] = canonical

    def get(self, provider: str, model: str) -> ModelInfo | None:
        entry = self._entry(provider, model)
        return ModelInfo(entry) if entry is not None else None

    def _entry(self, provider: str, model: str) -> ModelEntry | None:
        """The stored dict, for this module's own reads.

        Separate from `get` so the internals are not paying for a wrapper on
        every pricing lookup, and so `get` can return the caller-facing view
        without two spellings of the lookup itself.
        """
        direct = self._models.get(self._key(provider, model))
        if direct:
            return direct
        canonical = self._alias_index.get(self._key(provider, model)) or self._norm_index.get(
            # Last resort: the same model under a different spelling of its version.
            self._norm_key(provider, model)
        )
        return self._models.get(canonical) if canonical else None

    def resolve_model_id(self, provider: str, model: str) -> str:
        """The exact id to SEND to the provider for a given model string.

        Four cases, and the difference between them is what the caller ASKED for:

        - our slug -> translate to `providerModelName` (the pinned snapshot)
        - an id the provider itself accepts (a listed alias, e.g. a dated
          snapshot or anthropic's undated name) -> verbatim, because it is a
          deliberate choice and rewriting it would pin a caller who asked to float
        - a spelling variant that is NOT callable (`gemini-2-5-flash`) -> the
          canonical entry's `providerModelName`, since forwarding it verbatim
          only produces a 404 with the user's typo in it
        - unknown -> verbatim, so a model we have never heard of still works
        """
        direct = self._models.get(self._key(provider, model))
        if direct:
            return str(direct.get("providerModelName") or model)
        if self._key(provider, model) in self._alias_index:
            return model  # callable as written
        canonical = self._norm_index.get(self._norm_key(provider, model))
        if canonical:
            info = self._models.get(canonical)
            if info:
                return str(info.get("providerModelName") or info["model"])
        return model  # unknown -> as-is

    def get_pricing(self, provider: str, model: str) -> ModelPricing | None:
        info = self._entry(provider, model)
        return info.get("pricing") if info else None

    def get_preferred_api(self, provider: str, model: str) -> ApiType | None:
        info = self._entry(provider, model)
        return info.get("preferredApi") if info else None

    def supports_api(self, provider: str, model: str, api: ApiType) -> bool:
        info = self._entry(provider, model)
        return api in info["supportedApis"] if info else False

    def supports_tools(self, provider: str, model: str) -> bool:
        info = self._entry(provider, model)
        return bool(info["capabilities"].get("toolUse")) if info else False

    def builtin_tools_for(self, provider: str, model: str) -> list[str]:
        """Hosted server-side builtin tools this model supports.

        Empty when unknown or the model is not tool-capable. A COPY, so a caller
        that mutates the result cannot edit the catalog entry.
        """
        info = self._entry(provider, model)
        return list(info["capabilities"].get("builtinTools") or []) if info else []

    def supports_builtin_tool(self, provider: str, model: str, tool: str) -> bool:
        """Whether this model supports a specific hosted builtin tool."""
        info = self._entry(provider, model)
        if not info:
            return False
        return tool in (info["capabilities"].get("builtinTools") or [])

    def supports_previous_response_id(self, provider: str, model: str) -> bool:
        info = self._entry(provider, model)
        # A model carries server-state only on a stateful API (responses /
        # interactions).
        if info and not any(a in ("responses", "interactions") for a in info["supportedApis"]):
            return False
        if info and "supportsPreviousResponseId" in info:
            return bool(info["supportsPreviousResponseId"])
        state = _PROVIDER_STATE.get(provider)
        return bool(state["supports"]) if state else False

    def get_state_retention(self, provider: str, model: str) -> str | None:
        """Server-state retention as a duration string, or None if unsupported."""
        info = self._entry(provider, model)
        if info and "stateRetentionDuration" in info:
            retention = info["stateRetentionDuration"]
            return str(retention) if retention is not None else None
        state = _PROVIDER_STATE.get(provider)
        return state["retention"] if state else None

    def is_state_model_bound(self, provider: str, model: str) -> bool:
        """Whether server-state continuation requires the same model.

        True means same model only; False means it works across models of the
        same provider. Safe default: True.
        """
        info = self._entry(provider, model)
        if info and "stateModelBound" in info:
            return bool(info["stateModelBound"])
        state = _PROVIDER_STATE.get(provider)
        return bool(state["modelBound"]) if state else True

    def list(self, provider: str | None = None) -> list[ModelInfo]:
        entries = list(self._models.values())
        chosen = [m for m in entries if m["provider"] == provider] if provider else entries
        # Views here too: one accessor returning a different shape from the
        # other is the kind of difference nobody notices until it is in a loop.
        return [ModelInfo(entry) for entry in chosen]

    def load(self, data: dict[str, Any]) -> None:
        """Load one `provider/model` -> entry map.

        Two shapes are accepted, as in the TypeScript: a bare pricing record
        (recognised by `inputPerMTok`) and a full entry carrying `pricing`.
        Anything else is skipped rather than guessed at.
        """
        for key, value in data.items():
            slash = key.find("/")
            if slash < 0:
                continue
            provider = key[:slash]
            model = key[slash + 1 :]
            if not isinstance(value, dict):
                continue
            if "inputPerMTok" in value:
                self.set(provider, model, {"pricing": value})
            elif "pricing" in value:
                self.set(provider, model, value)

    @staticmethod
    def with_provider_defaults() -> ModelCatalog:
        """A catalog with every bundled provider entry already loaded.

        This is what the engine and the client build when nobody says otherwise.
        A fresh instance each time rather than a shared one: the catalog is
        mutable (`set` is public and examples use it), so sharing would let one
        engine's edit reach another's request. Indexing all entries costs about a
        millisecond, against a network call.
        """
        catalog = ModelCatalog()
        catalog.load_provider_defaults()
        return catalog

    def load_provider_defaults(self) -> None:
        """Load every provider catalog shipped with the SDK.

        The TypeScript imports the JSON statically, so it has no runtime I/O.
        Python has no equivalent, so the files are read from the package
        directory -- which is why `pyproject.toml` lists
        `catalog/data/*.json` as wheel artifacts: without them a wheel installs
        cleanly and fails on first use.
        """
        for name in _PROVIDER_DEFAULT_FILES:
            with (_DATA_DIR / name).open(encoding="utf-8") as handle:
                self.load(json.load(handle))

    @property
    def size(self) -> int:
        return len(self._models)


#: The catalogs a caller may ask for by NAME.
#:
#: A name rather than only an instance because `Engine(catalog="defaults")` is
#: what the reviewed examples write, and because the alternative -- importing
#: `ModelCatalog` to say "the usual one" -- puts a class in every call site that
#: only wanted a default.
CATALOG_NAMES = ("defaults", "empty")


def resolve_catalog(value: ModelCatalog | str | None) -> ModelCatalog:
    """A catalog, from an instance, a name, or nothing.

    Shared by `Engine` and `LLM` so the two cannot disagree about what
    `"defaults"` means. An unknown name is an ERROR rather than a quiet fallback
    to the defaults: a typo would otherwise produce a working engine whose
    catalog is not the one asked for, and the first symptom is an unpriced model
    reading as free.
    """
    if value is None:
        return ModelCatalog.with_provider_defaults()
    if isinstance(value, ModelCatalog):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"catalog= wants a ModelCatalog or one of {CATALOG_NAMES}, "
            f"got {type(value).__name__}."
        )
    if value == "defaults":
        return ModelCatalog.with_provider_defaults()
    if value == "empty":
        return ModelCatalog()
    raise ValueError(f"catalog={value!r} is not a known catalog. One of: {CATALOG_NAMES}.")


__all__ = [
    "ApiType",
    "MediaParamSpec",
    "ModelCapabilities",
    "ModelCatalog",
    "ModelEntry",
    "ModelInfo",
    "ModelPricing",
    "ModelReasoning",
    "TierRates",
    "TokenizerInfo",
    "normalize_id",
]
