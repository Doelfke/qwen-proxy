"""Normalize a non-streaming chat-completions response body.

Covers both directions of the problem: tool calls that leaked into ``content``
as XML (recovered by :mod:`qwen_proxy.recovery`), and tool calls that were
already structured but carry mangled argument values (collapsed URLs, broken
command strings).
"""

from __future__ import annotations

import json
import logging

from .repair import repair_command, repair_url
from .recovery import normalize_message

log = logging.getLogger("toolcall_proxy")


def normalize_response_body(body: bytes) -> bytes:
    """Normalize one non-streaming chat-completions response.

    Also repairs ``tool_calls`` entries that were already structured but carry mangled
    argument values (collapsed URLs, broken command strings).
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        log.debug("non-stream body is not JSON (%d bytes); passed through", len(body))
        return body
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        return body
    total = 0
    dirty = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            sub = choice.get(key)
            if not isinstance(sub, dict):
                continue
            total += normalize_message(sub)
            dirty = repair_structured_calls(sub) or dirty
    if total or dirty:
        log.info(
            "normalized %d recovered call(s) in a non-stream response", total
        )
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")
    log.debug("non-stream response unmodified (%d bytes)", len(body))
    return body


def repair_structured_calls(msg: dict) -> bool:
    """Repair values inside already-structured tool_calls (URLs, commands).

    Returns True if any argument value was changed, so callers can tell whether
    the (parsed) payload must be re-serialized. A repair that happens without this
    would otherwise be silently discarded by ``normalize_response_body``.
    """
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return False
    changed = False
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        raw = fn.get("arguments")
        if not isinstance(raw, str):
            continue
        try:
            args = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(args, dict):
            continue
        fixed = False
        for key in list(args):
            if not isinstance(args[key], str):
                continue
            if key == "url":
                new = repair_url(args[key])
            elif key == "command":
                new = repair_command(args[key])
            else:
                continue
            if new != args[key]:
                args[key] = new
                fixed = True
        if fixed:
            fn["arguments"] = json.dumps(args, ensure_ascii=False)
            changed = True
    return changed
