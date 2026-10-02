"""Anthropic's code-execution container, and the skills loaded into it.

Transposed from `unified-library-ts/src/llm/providers/anthropic/container.ts`.

The container is the sandbox Anthropic's code execution tool runs in. It used to be
invisible: we sent no `container` and dropped the one the response carried, so a
caller could neither reuse a warm container nor load a skill into it.

Measured 2026-10-02, all on a plain key with NO beta header (the param is GA):

- A turn that runs code answers with `container: {id, expires_at}` -- about five
  minutes of life, so reuse is a real option rather than a theoretical one.
- Passing `{"id": ...}` back REUSES it: the same id comes out again.
- `{"skills": [{"type": "anthropic", "skill_id": "xlsx"}]}` is accepted and
  VALIDATED server-side -- a skill that does not exist is a `400 Unknown Anthropic
  skill`, not a silent ignore, so a typo fails loudly. `GET /v1/skills` lists the
  built-in ones (`xlsx`, `pptx`, `pdf`, ...) and is GA too.
- A requested `version: "latest"` comes back RESOLVED (`"20260914"`), which is why
  the response's version is worth reporting rather than echoing the request.
- Asking for skills on a turn that never runs code answers `container: null`: the
  container is created when it is needed, not when it is requested.

And the one shape a reasonable guess gets wrong: when STREAMING, `message_start`
carries `container: null` and the real container arrives on
`message_delta.delta.container`. Reading the opening frame would report `None` for
every streamed turn.

camelCase in the public API, snake_case on the wire -- as everywhere in this port.
A skill is `{"type", "skillId", "version"?}` to a caller and
`{"type", "skill_id", "version"?}` on the wire; the container reports `expiresAt`
and the wire sends `expires_at`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: `AnthropicSkillRef` -- a skill to load. `skillId` is the skill's own id
#: (`"xlsx"`, or a custom skill's id) and `version` accepts `"latest"`.
AnthropicSkillRef = dict[str, Any]

#: `AnthropicContainerRequest` -- `{id?, skills?}`: reuse a container, load skills
#: into it, or both.
AnthropicContainerRequest = dict[str, Any]

#: `ContainerInfo` -- `{id, expiresAt, skills?}`: the container a turn actually
#: used, as reported back, with skill versions RESOLVED.
ContainerInfo = dict[str, Any]


def to_wire_container(req: Mapping[str, Any]) -> dict[str, Any]:
    """`providerOptions["container"]` -> the wire's `container`.

    Only the keys the caller set: sending `skills: []` or a null `id` would be a
    request for something they did not ask for.
    """
    out: dict[str, Any] = {}
    if req.get("id"):
        out["id"] = req["id"]
    skills = req.get("skills")
    if isinstance(skills, list) and skills:
        out["skills"] = [_wire_skill(skill) for skill in skills]
    return out


def _wire_skill(skill: Any) -> dict[str, Any]:
    """One skill ref, snake_cased, or a clear error.

    A malformed entry RAISES rather than being skipped. Skipping it would load no
    skill while the caller believes one is loaded -- the same silent-success failure
    the shell tool's warning exists to prevent -- and letting it through produces a
    server-side 400 about a field the caller never wrote. Both are worse than saying
    so here, where the mistake is.
    """
    if not isinstance(skill, Mapping):
        raise TypeError(
            f"container.skills entries must be {{'type', 'skillId'}} mappings; got {skill!r}"
        )
    kind = skill.get("type")
    skill_id = skill.get("skillId")
    if kind not in ("anthropic", "custom"):
        raise ValueError(
            f"container.skills[].type must be 'anthropic' or 'custom'; got {kind!r}"
        )
    if not isinstance(skill_id, str) or not skill_id:
        raise ValueError(f"container.skills[].skillId must be a non-empty string; got {skill_id!r}")
    entry: dict[str, Any] = {"type": kind, "skill_id": skill_id}
    if skill.get("version"):
        entry["version"] = skill["version"]
    return entry


def container_from_wire(raw: Any) -> ContainerInfo | None:
    """The wire's `container` -> `ContainerInfo`, or `None` when there was none.

    `null` is the normal answer for a turn that never ran code, so it is not an
    error and not worth a warning -- it means no container was created.
    """
    if not isinstance(raw, Mapping):
        return None
    if not isinstance(raw.get("id"), str):
        return None
    skills_raw = raw.get("skills")
    skills: list[AnthropicSkillRef] = []
    if isinstance(skills_raw, list):
        for s in skills_raw:
            if not isinstance(s, Mapping):
                continue
            entry: dict[str, Any] = {
                "type": "custom" if s.get("type") == "custom" else "anthropic",
                "skillId": s.get("skill_id") if isinstance(s.get("skill_id"), str) else "",
            }
            if isinstance(s.get("version"), str):
                entry["version"] = s["version"]
            skills.append(entry)
    expires = raw.get("expires_at")
    info: ContainerInfo = {
        "id": raw["id"],
        "expiresAt": expires if isinstance(expires, str) else "",
    }
    if skills:
        info["skills"] = skills
    return info


__all__ = [
    "AnthropicContainerRequest",
    "AnthropicSkillRef",
    "ContainerInfo",
    "container_from_wire",
    "to_wire_container",
]
