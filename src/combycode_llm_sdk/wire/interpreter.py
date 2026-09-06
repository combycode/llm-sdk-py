"""The wire interpreter: a spec is data, and this executes it.

Transposed from `unified-library-ts/src/wire/interpreter.ts`, function for
function. The specs themselves are shared byte-for-byte with the TypeScript
tree, so this file has to agree with that one about what each rule MEANS or the
same spec builds two different requests.

Two JavaScript semantics are reproduced deliberately, because Python's
equivalents differ in ways that reach the wire:

  `js_truthy`  JS says an empty array is truthy; Python says it is falsy. A
               spec guarded by `truthy: tools` with `tools: []` therefore emits
               the field in TypeScript and would omit it here.

  `js_string`  `String(true)` is "true" and `String(None)` is "null". Python's
               str() gives "True" and "None", which would ship a header value
               no provider recognises.

`MISSING` is separate from `None` for the same reason TypeScript separates
`undefined` from `null`: a request carrying an explicit JSON null is not the
same as one that omitted the field, and specs read the difference.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, KeysView, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, quote_plus

Json = Any

#: Absent, as distinct from a JSON `null` that is genuinely present.
MISSING: Any = object()

#: A template that evaluated to nothing. Distinct from `None`, which is a value
#: a spec may legitimately want to emit.
OMIT: Any = object()


def is_obj(v: Any) -> bool:
    """A JSON object -- a mapping, and specifically not a list."""
    return isinstance(v, Mapping)


def js_truthy(v: Any) -> bool:
    """JavaScript's `Boolean(v)`.

    The divergence that matters: `[]` and `{}` are truthy in JavaScript and
    falsy in Python, so a `truthy:` guard over an empty list decides the
    opposite way if this is left to Python.
    """
    if v is MISSING or v is None or v is False:
        return False
    if v is True:
        return True
    if isinstance(v, (int, float)):
        return not (v == 0 or (isinstance(v, float) and math.isnan(v)))
    if isinstance(v, str):
        return v != ""
    return True


def js_json(v: Any) -> str:
    """JavaScript's `JSON.stringify(v)`.

    Python's `json.dumps` puts a space after every `:` and `,`; JSON.stringify
    does not. That difference is invisible until a serialized value becomes part
    of the OUTPUT -- streamed tool-call `arguments` are a JSON string, and
    `{"city": "Paris"}` is not `{"city":"Paris"}`. Stated once, here, so no call
    site has to remember it.
    """
    return json.dumps(v, separators=(",", ":"))


def js_string(v: Any) -> str:
    """JavaScript's `String(v)`, for values that reach a header or a URL."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None or v is MISSING:
        return "null"
    if isinstance(v, float) and v.is_integer() and not math.isinf(v):
        return str(int(v))
    if isinstance(v, (Mapping, list, tuple)):
        return js_json(v)
    return str(v)


def _is_index(part: str) -> bool:
    """ASCII digits only. `str.isdigit()` is true for other numerals too, and
    `int("٥")` is 5 -- which would index an array from a character the
    provider never sent."""
    return part.isascii() and part.isdigit()


def get_path(root: Any, path: str) -> Any:
    if not path:
        return root
    cur = root
    for part in path.split("."):
        if cur is MISSING or cur is None:
            return MISSING
        if isinstance(cur, Mapping):
            cur = cur.get(part, MISSING)
        elif isinstance(cur, (list, tuple)):
            # A JavaScript array IS an object, so `arr["0"]` is `arr[0]` and the
            # TypeScript `cur[part]` resolves a numeric segment without knowing
            # it is one. Python has no such coercion: without this branch a spec
            # path like `raw.choices.0.message` silently yields MISSING and the
            # field is simply absent from the built value -- no error, no clue.
            #
            # Latent until now: not one of the 150 request specs uses a numeric
            # segment. Seven of the response and stream specs do.
            cur = cur[int(part)] if _is_index(part) and int(part) < len(cur) else MISSING
        else:
            cur = getattr(cur, part, MISSING)
    return cur


def set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = root
    for p in parts[:-1]:
        if not is_obj(cur.get(p)):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def delete_path(root: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    cur: Any = root
    for p in parts[:-1]:
        cur = cur.get(p) if isinstance(cur, Mapping) else None
        if not is_obj(cur):
            return
    cur.pop(parts[-1], None)


def deep_merge(target: Any, source: Any) -> Any:
    if not is_obj(target) or not is_obj(source):
        return source
    out = dict(target)
    for k, v in source.items():
        out[k] = deep_merge(out[k], v) if is_obj(v) and is_obj(out.get(k)) else v
    return out


# ── registry ────────────────────────────────────────────────────────────────

Transform = Callable[..., Json]
Builder = Callable[..., Json]
Predicate = Callable[..., bool]
Effect = Callable[..., None]


@dataclass(slots=True)
class Registry:
    transforms: dict[str, Transform] = field(default_factory=dict)
    builders: dict[str, Builder] = field(default_factory=dict)
    predicates: dict[str, Predicate] = field(default_factory=dict)
    effects: dict[str, Effect] = field(default_factory=dict)


@dataclass(slots=True)
class MultipartField:
    name: str
    kind: str
    value: Json = None


@dataclass(slots=True)
class Item:
    value: Any
    index: int
    is_last: bool


@dataclass(slots=True)
class Ctx:
    req: Any
    spec: Mapping[str, Any]
    flavor: str
    #: Adapter-level configuration (baseURL, apiKey, ...) referenced by `$config`.
    #: Keeps the URL declarative rather than pushing it into a named transform.
    config: dict[str, Any]
    variants: Any
    body: dict[str, Any]
    #: Per-item scope while inside a $map.
    item: Item | None = None
    #: Collected multipart fields, when the spec declares a multipart body.
    multipart: list[MultipartField] | None = None
    #: What the build decided to leave out, and why -- surfaced to the caller.
    notes: list[str] = field(default_factory=list)

    def with_item(self, value: Any, index: int, is_last: bool) -> Ctx:
        """A copy scoped to one $map item. Shares `body` and `notes` on purpose:
        an effect fired inside a map still writes to the request being built."""
        return Ctx(
            req=self.req, spec=self.spec, flavor=self.flavor, config=self.config,
            variants=self.variants, body=self.body, item=Item(value, index, is_last),
            multipart=self.multipart, notes=self.notes,
        )


@dataclass(slots=True)
class BuiltRequest:
    body: dict[str, Any] = field(default_factory=dict)
    #: Anything the spec deliberately left out, and why -- e.g. a hosted tool
    #: this provider will not run beside the attached content. The runtime turns
    #: these into `on_warning`; they are never silent.
    notes: list[str] | None = None
    headers: dict[str, str] | None = None
    path: str | None = None
    url: str | None = None
    method: str | None = None
    #: The body is caller-supplied bytes (bodyKind 'raw').
    raw_body: bool = False
    #: Present instead of a JSON body when bodyKind is 'multipart'.
    multipart: list[MultipartField] | None = None
    #: The body is form-urlencoded: `body` holds the FIELDS and the caller
    #: encodes them. Same split as multipart -- the spec says what the form
    #: carries, the runtime does the encoding, and a frozen fixture stays
    #: readable as fields rather than as one percent-escaped string.
    form_body: bool = False
    #: True when the spec declares the request carries no body at all.
    no_body: bool = False


# ── condition evaluation ────────────────────────────────────────────────────


def eval_cond(cond: Mapping[str, Any] | None, ctx: Ctx, reg: Registry) -> bool:
    if not cond:
        return True
    c = cond

    if "defined" in c:
        return get_path(ctx.req, c["defined"]) is not MISSING
    if "truthy" in c:
        return js_truthy(get_path(ctx.req, c["truthy"]))
    if "eq" in c:
        return bool(get_path(ctx.req, c["eq"][0]) == c["eq"][1])
    if "ne" in c:
        return bool(get_path(ctx.req, c["ne"][0]) != c["ne"][1])
    if "variant" in c:
        return c["variant"] in ctx.variants
    if "flavor" in c:
        f = c["flavor"]
        return ctx.flavor in f if isinstance(f, list) else ctx.flavor == f
    if "pred" in c:
        p = reg.predicates.get(c["pred"])
        if p is None:
            raise ValueError(f"unknown predicate: {c['pred']}")
        return p(ctx)
    if "itemTruthy" in c:
        return js_truthy(get_path(ctx.item.value if ctx.item else MISSING, c["itemTruthy"]))
    if "itemEq" in c:
        return ctx.item is not None and ctx.item.value == c["itemEq"]
    if "isLast" in c:
        return ctx.item is not None and ctx.item.is_last is True
    if "isFunctionTool" in c:
        return reg.predicates["isFunctionTool"](ctx)
    if "builtin" in c:
        t = ctx.item.value if ctx.item else None
        return not reg.predicates["isFunctionTool"](ctx) and _type_of(t) == c["builtin"]
    if "hasPartType" in c:
        messages = _as_list(get_path(ctx.req, "messages"))
        return any(
            any(_type_of(part) in c["hasPartType"] for part in m["content"])
            for m in messages
            if isinstance(m, Mapping) and isinstance(m.get("content"), list)
        )
    if "hasTool" in c:
        tools = _as_list(get_path(ctx.req, "tools"))
        present = any(
            not reg.predicates["isFunctionTool"](ctx.with_item(t, 0, False))
            and _type_of(t) == c["hasTool"]
            for t in tools
        )
        if not present:
            return False
        # Present, but this provider may refuse it alongside what else is in the
        # request. Reported as absent so the spec's own guard omits it, and
        # recorded so the runtime can tell the caller rather than leaving them
        # to wonder.
        for tc in ctx.spec.get("toolConstraints") or []:
            if tc.get("tool") == c["hasTool"] and eval_cond(tc.get("conflictsWith"), ctx, reg):
                if tc["why"] not in ctx.notes:
                    ctx.notes.append(tc["why"])
                return False
        return True
    if "hasFunctionTool" in c:
        tools = _as_list(get_path(ctx.req, "tools"))
        return any(
            reg.predicates["isFunctionTool"](ctx.with_item(t, 0, False)) for t in tools
        )
    if "includes" in c:
        arr = get_path(ctx.req, c["includes"][0])
        return isinstance(arr, list) and c["includes"][1] in arr
    if "not" in c:
        return not eval_cond(c["not"], ctx, reg)
    if "all" in c:
        return all(eval_cond(x, ctx, reg) for x in c["all"])
    if "any" in c:
        return any(eval_cond(x, ctx, reg) for x in c["any"])

    raise ValueError(f"unknown condition: {json.dumps(dict(c), default=str)}")


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _type_of(v: Any) -> Any:
    return v.get("type") if isinstance(v, Mapping) else None


# ── template evaluation ─────────────────────────────────────────────────────


def eval_template(tpl: Json, ctx: Ctx, reg: Registry) -> Json:
    if isinstance(tpl, list):
        out: list[Json] = []
        for el in tpl:
            # `$each` splices an evaluated array INTO this array -- the array
            # analogue of `$spread`. Needed wherever a `$map` has to sit beside
            # literal entries: a bare `$map` there nests one level down, and a
            # nested array is a different request, not a formatting detail.
            if is_obj(el) and "$each" in el:
                many = eval_template(el["$each"], ctx, reg)
                if many is not OMIT and isinstance(many, list):
                    out.extend(many)
                continue
            v = eval_template(el, ctx, reg)
            if v is not OMIT:
                out.append(v)
        return out

    if not is_obj(tpl):
        return tpl

    t: Mapping[str, Any] = tpl

    # Conditional value: emit `$value` only when `$when` holds.
    if "$when" in t:
        if not eval_cond(t["$when"], ctx, reg):
            return OMIT
        return eval_template(t["$value"], ctx, reg) if "$value" in t else OMIT

    # Read a request path (or the current $map item with a leading `@`).
    if "$" in t:
        p: str = t["$"]
        raw = (
            get_path(ctx.item.value if ctx.item else MISSING, p[1:])
            if p.startswith("@")
            else get_path(ctx.req, p)
        )
        v = t.get("$default", MISSING) if raw is MISSING else raw
        return OMIT if v is MISSING else v

    # Adapter config value (baseURL, apiKey).
    if "$config" in t:
        v = ctx.config.get(t["$config"], MISSING)
        return OMIT if v is MISSING else v

    # String built from parts; the only way to concatenate, kept deliberately dull.
    if "$join" in t:
        parts = [eval_template(x, ctx, reg) for x in t["$join"]]
        if any(p is OMIT for p in parts):
            return OMIT
        sep = "" if t.get("$sep") is None else js_string(t["$sep"])
        return sep.join(js_string(p) for p in parts)

    # Table lookup keyed by a request path.
    if "$table" in t:
        table = (ctx.spec.get("tables") or {}).get(t["$table"])
        if not table:
            raise ValueError(f"unknown table: {t['$table']}")
        key = js_string(get_path(ctx.req, t["$key"]))
        v = table[key] if key in table else t.get("$default", MISSING)
        return OMIT if v is MISSING else v

    # Named transform over the whole request (or the current item with `$arg`).
    if "$call" in t:
        fn = reg.transforms.get(t["$call"])
        if fn is None:
            raise ValueError(f"unknown transform: {t['$call']}")
        arg = MISSING if "$arg" not in t else eval_template(t["$arg"], ctx, reg)
        v = fn(None if arg is OMIT or arg is MISSING else arg, ctx)
        return OMIT if v is None or v is MISSING else v

    # Map over a request array with per-item cases.
    if "$map" in t:
        found = get_path(ctx.req, t["$map"])
        arr = found if isinstance(found, list) else t.get("$mapDefault")
        if not isinstance(arr, list):
            return OMIT
        out_items: list[Json] = []
        for i, raw_item in enumerate(arr):
            item_ctx = ctx.with_item(raw_item, i, i == len(arr) - 1)
            matched = False
            for kase in t["$case"]:
                if not eval_cond(kase.get("when"), item_ctx, reg):
                    continue
                v = eval_template(kase["value"], item_ctx, reg)
                if v is not OMIT:
                    out_items.append(v)
                matched = True
                break
            if not matched and not t.get("$dropUnmatched"):
                raise ValueError(
                    f"$map item {i} matched no case and $dropUnmatched is not set"
                )
        return out_items

    # Plain object: evaluate each value, honouring `$spread` and omission.
    out_obj: dict[str, Any] = {}
    for k, v in t.items():
        # Comments inside a template. Without a reserved form, a plain `note`
        # key silently ships to the provider -- which it did, on the first run.
        if k == "$note":
            continue
        if k == "$spread":
            src = eval_template(v, ctx, reg)
            if src is not OMIT and is_obj(src):
                out_obj.update(src)
            continue
        val = eval_template(v, ctx, reg)
        if val is not OMIT:
            out_obj[k] = val
    return out_obj


# ── model variant resolution ────────────────────────────────────────────────


def resolve_variants(spec: Mapping[str, Any], model: str, reg: Registry) -> KeysView[str]:
    """The flags a spec's variant rules set for this model.

    Returns a keys view rather than a `set` because TypeScript's `Set` preserves
    insertion order and Python's does not -- and the order is observable, since
    callers iterate it.
    """
    flags: dict[str, None] = {}
    ident = re.sub(r"^[a-z]+/", "", model.lower())
    variants = spec.get("variants") or []

    for v in variants:
        if v.get("idMatch") and re.search(v["idMatch"], ident):
            flags[v["flag"]] = None
        if v.get("fn"):
            fn = reg.transforms.get(v["fn"])
            if fn is None:
                raise ValueError(f"unknown variant fn: {v['fn']}")
            if fn(model, Ctx(req={"model": model}, spec=spec, flavor="", config={},
                             variants={}, body={})):
                flags[v["flag"]] = None

    # `unless` lets a spec express "this flag applies except when that one did".
    for v in variants:
        if v.get("unless") and v["unless"] in flags:
            flags.pop(v["flag"], None)

    return flags.keys()


# ── the interpreter ─────────────────────────────────────────────────────────


def _model_of(req: Any) -> str:
    """`req.model ?? ''` -- nullish, not falsy.

    `MISSING or ""` returns the sentinel, because a bare object is truthy in
    Python. TypeScript's `??` only falls through on null/undefined, and the
    difference surfaces the moment a spec is built from a request with no model.
    """
    model = get_path(req, "model") if req is not None else MISSING
    return "" if model is MISSING or model is None else js_string(model)


def build_from_spec(
    spec: Mapping[str, Any],
    req: Any,
    reg: Registry,
    flavor: str | None = None,
    #: Coverage hook: called with every rule that actually fired.
    on_use: Callable[[str, str], None] | None = None,
    #: Adapter config exposed to `$config`.
    config: dict[str, Any] | None = None,
) -> BuiltRequest:
    flavor = spec.get("provider", "") if flavor is None else flavor
    envelope: Mapping[str, Any] = spec.get("envelope") or {}
    ctx = Ctx(
        req=req,
        spec=spec,
        flavor=flavor,
        config=config or {},
        variants=resolve_variants(spec, _model_of(req), reg),
        body={},
    )

    def used(kind: str, name: str) -> None:
        if on_use is not None:
            on_use(kind, name)

    for v in ctx.variants:
        used("variant", v)

    # 1. fields
    for f in spec.get("fields") or []:
        raw = get_path(req, f["from"])
        present = js_truthy(raw) if f.get("presence", "defined") == "truthy" else raw is not MISSING
        value = raw if present else f.get("default", MISSING)
        if value is MISSING:
            continue
        if not present and f.get("default", MISSING) is MISSING:
            continue
        if not eval_cond(f.get("when"), ctx, reg):
            continue

        out_value: Json = value
        if f.get("table"):
            table = (spec.get("tables") or {}).get(f["table"])
            if not table:
                raise ValueError(f"unknown table: {f['table']}")
            key = js_string(value)
            out_value = table[key] if key in table else f.get("tableDefault", MISSING)
            if out_value is MISSING:
                continue
        if f.get("call"):
            fn = reg.transforms.get(f["call"])
            if fn is None:
                raise ValueError(f"unknown transform: {f['call']}")
            out_value = fn(value, ctx)
            if out_value is None or out_value is MISSING:
                continue
        set_path(ctx.body, f["to"], out_value)
        used("field", f["to"])

    # 2. blocks, in declaration order (output_config merge order depends on it)
    for b in spec.get("blocks") or []:
        if not eval_cond(b.get("when"), ctx, reg):
            continue
        if b.get("call"):
            fn = reg.builders.get(b["call"])
            if fn is None:
                raise ValueError(f"unknown builder: {b['call']}")
            value = fn(ctx)
        else:
            value = eval_template(b.get("template"), ctx, reg)
        if value is OMIT or value is None or value is MISSING:
            continue

        if not b.get("to"):
            if is_obj(value):
                ctx.body.update(value)
        elif b.get("merge"):
            existing = get_path(ctx.body, b["to"])
            set_path(ctx.body, b["to"], deep_merge({} if existing is MISSING else existing, value))
        else:
            set_path(ctx.body, b["to"], value)

        used("block", b["name"])

        for e in b.get("effects") or []:
            fn_e = reg.effects.get(e)
            if fn_e is None:
                raise ValueError(f"unknown effect: {e}")
            fn_e(ctx)

    # 2b. multipart body, described as fields rather than a blob
    if envelope.get("bodyKind") == "multipart":
        fields: list[MultipartField] = []
        for f in spec.get("multipart") or []:
            if not eval_cond(f.get("when"), ctx, reg):
                continue
            if f.get("file"):
                fields.append(MultipartField(name=f["name"], kind="file"))
            else:
                v = eval_template(f.get("value"), ctx, reg)
                if v is OMIT or v is None or v is MISSING:
                    pass  # nothing to append
                elif f.get("repeat") and isinstance(v, list):
                    for item in v:
                        fields.append(MultipartField(name=f["name"], kind="value", value=item))
                else:
                    fields.append(MultipartField(name=f["name"], kind="value", value=v))
            used("block", f"multipart:{f['name']}")
        ctx.multipart = fields

    # 3. per-flavor overlay
    for op in ((spec.get("overlays") or {}).get(flavor) or {}).get("ops") or []:
        if not eval_cond(op.get("when"), ctx, reg):
            continue
        used("overlay", f"{flavor}:{op['op']}:{op.get('from') or op.get('to') or op.get('call')}")
        kind = op["op"]
        if kind == "rename":
            v = get_path(ctx.body, op["from"])
            if js_truthy(v):
                set_path(ctx.body, op["to"], v)
                delete_path(ctx.body, op["from"])
        elif kind == "delete":
            delete_path(ctx.body, op["from"])
        elif kind == "set":
            v = eval_template(op.get("value"), ctx, reg)
            if v is not OMIT:
                set_path(ctx.body, op["to"], v)
        elif kind == "mergeFrom":
            src = get_path(ctx.req, op["from"])
            if is_obj(src):
                ctx.body.update(src)
        elif kind == "call":
            fn_e = reg.effects.get(op["call"])
            if fn_e is None:
                raise ValueError(f"unknown overlay effect: {op['call']}")
            fn_e(ctx)

    # 4. envelope
    out = BuiltRequest(body=ctx.body)
    if ctx.notes:
        out.notes = ctx.notes
    if ctx.multipart is not None:
        out.multipart = ctx.multipart
    if envelope.get("bodyKind") == "none":
        out.no_body = True
    # `raw`: the caller attaches the bytes; say so rather than emitting an empty body.
    if envelope.get("bodyKind") == "raw":
        out.raw_body = True
    if envelope.get("bodyKind") == "form":
        out.form_body = True

    headers: dict[str, str] = {}
    for h in envelope.get("headers") or []:
        if not eval_cond(h.get("when"), ctx, reg):
            continue
        if h.get("spread") is not None:
            many = eval_template(h["spread"], ctx, reg)
            if many is not OMIT and is_obj(many):
                for k, v in many.items():
                    if v is None or v is MISSING:
                        continue
                    headers[k] = js_string(v)
                    used("header", k)
            continue
        if h.get("name") is None:
            raise ValueError(f"{spec.get('id')}: a header needs either a name or a spread")
        v = eval_template(h.get("value"), ctx, reg)
        if v is not OMIT and v is not None and v is not MISSING:
            headers[h["name"]] = js_string(v)
            used("header", h["name"])
    if envelope.get("headers") is not None:
        out.headers = headers

    if envelope.get("method"):
        out.method = envelope["method"]
    if envelope.get("url") is not None:
        u = eval_template(envelope["url"], ctx, reg)
        if u is not OMIT and u is not None and u is not MISSING:
            out.url = js_string(u)
    if envelope.get("path") is not None:
        p = eval_template(envelope["path"], ctx, reg)
        if p is not OMIT and p is not None and p is not MISSING:
            out.path = js_string(p)

    # 5. query parameters, each able to drop out on its own.
    params: list[str] = []
    for q in envelope.get("query") or []:
        if not eval_cond(q.get("when"), ctx, reg):
            continue
        v = eval_template(q.get("value"), ctx, reg)
        if v is OMIT or v is None or v is MISSING:
            continue
        esc = _form_escape if envelope.get("queryEncoding") == "form" else _uri_escape
        params.append(f"{esc(q['name'])}={esc(js_string(v))}")
        used("header", f"query:{q['name']}")
    if params:
        target = "url" if out.url is not None else "path"
        base = getattr(out, target)
        if base is None:
            raise ValueError(f"{spec.get('id')}: query parameters with no url or path to attach to")
        joiner = "&" if "?" in base else "?"
        setattr(out, target, f"{base}{joiner}{'&'.join(params)}")

    return out


# ── realtime: a session is three artifacts, not one request ─────────────────
#
# A WebSocket session has no request/response pair to build. It has a connection
# descriptor, a handshake frame, and the frames of each turn -- and the specs
# describe all three, so the same JSON drives both libraries here too.


@dataclass(frozen=True)
class BuiltConnection:
    """Where to open the socket, and under which subprotocols.

    OpenAI carries its API key in a SUBPROTOCOL rather than a header, because a
    browser cannot set a header on a WebSocket handshake -- which is why
    `protocols` is modelled at all.
    """

    url: str
    protocols: list[str] | None = None


def _op_ctx(spec: Mapping[str, Any], req: Any, config: Mapping[str, Any] | None) -> Ctx:
    """The context an operation is evaluated in.

    No variants and no body: an operation is evaluated whole from its own
    template, where a request is assembled field by field.
    """
    return Ctx(
        req=req,
        spec=spec,
        flavor=spec.get("provider", ""),
        config=dict(config or {}),
        variants=set(),
        body={},
    )


def _operation(spec: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    op = (spec.get("operations") or {}).get(operation)
    if op is None or not isinstance(op, Mapping):
        raise KeyError(f"spec {spec.get('id')!r} has no operation {operation!r}")
    found: Mapping[str, Any] = op
    return found


def build_connection(
    spec: Mapping[str, Any],
    operation: str,
    req: Any,
    reg: Registry,
    config: Mapping[str, Any] | None = None,
) -> BuiltConnection:
    """The connection descriptor for an operation (realtime `connect`)."""
    op = _operation(spec, operation)
    ctx = _op_ctx(spec, req, config)
    url = eval_template(op.get("url"), ctx, reg)
    if url is OMIT or url is None:
        raise ValueError(f"operation {operation!r} produced no url")
    protocols = None
    if op.get("protocols") is not None:
        value = eval_template(op["protocols"], ctx, reg)
        if value is not OMIT and isinstance(value, list):
            protocols = [js_string(p) for p in value]
    return BuiltConnection(url=js_string(url), protocols=protocols)


def build_frames(
    spec: Mapping[str, Any],
    operation: str,
    req: Any,
    reg: Registry,
    config: Mapping[str, Any] | None = None,
) -> list[Any]:
    """The outbound frames for an operation (realtime `open` / `send`).

    A list, because one turn can be several frames: OpenAI needs a second frame
    to ask for a reply where Gemini carries the same meaning as a field.
    """
    op = _operation(spec, operation)
    ctx = _op_ctx(spec, req, config)
    out: list[Any] = []
    for frame in op.get("frames") or []:
        if not eval_cond(frame.get("when"), ctx, reg):
            continue
        value = eval_template(frame.get("template"), ctx, reg)
        if value is not OMIT and value is not None:
            out.append(value)
    return out


def _uri_escape(s: str) -> str:
    """`encodeURIComponent`: everything but the unreserved set and `!~*'()`."""
    return quote(s, safe="-_.!~*'()")


def _form_escape(s: str) -> str:
    """`new URLSearchParams(...).toString()`: form encoding, space as `+`."""
    return quote_plus(s, safe="*").replace("~", "%7E")


__all__ = [
    "MISSING",
    "OMIT",
    "BuiltConnection",
    "BuiltRequest",
    "Ctx",
    "Item",
    "MultipartField",
    "Registry",
    "build_connection",
    "build_frames",
    "build_from_spec",
    "deep_merge",
    "delete_path",
    "eval_cond",
    "eval_template",
    "get_path",
    "js_json",
    "js_string",
    "js_truthy",
    "resolve_variants",
    "set_path",
]
