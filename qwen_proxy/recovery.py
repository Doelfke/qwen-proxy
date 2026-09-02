"""Recover tool calls that a model emitted as text inside ``content``.

Qwen-class models served through an OpenAI-compatible endpoint (vLLM/SGLang/...)
occasionally emit tool calls as *text* — the native ``<tool_calls>`` XML format
leaking through — instead of as structured ``tool_calls`` entries. This module
parses that XML out of a content string and turns it into OpenAI-shaped
``tool_calls`` dicts, repairing each argument value on the way.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from .repair import clean_parameter

log = logging.getLogger("toolcall_proxy")

_TOOL_CALLS_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>"
    r"|< tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

_INVOKE_RE = re.compile(
    r"<invoke>?\s*name=['\"]([^'\"]+)['\"][^>]*?/?>"
    r"(.*?)(?:</invoke>|(?=<invoke)|\Z)",
    re.DOTALL | re.IGNORECASE,
)
# A paired <parameter ... name=\"x\">value</parameter>. Same stray-'>' tolerance;
# group(1) = parameter name, group(2) = value (the tag body).
_PARAM_PAIR_RE = re.compile(
    r"<parameter>?\s*name=['\"]([^'\"]+)['\"][^>]*?>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
# A self-closing <parameter ... name=\"x\" value=\"y\"/> where the value is an attribute
# rather than the tag body. group(1) = parameter name, group(2) = value.
_PARAM_SELF_RE = re.compile(
    r"<parameter>?\s*name=['\"]([^'\"]+)['\"]\s*value=['\"]([^'\"]*)['\"][^>]*/>",
    re.DOTALL | re.IGNORECASE,
)


def invoke_args(body: str) -> dict:
    """Extract ``{name: value}`` for every parameter inside an ``<invoke>`` body.

    Reads both the paired form (``<parameter name="k">value</parameter>``) and the
    self-closing form (``<parameter name="k" value="v"/>``), tolerating the stray
    ``>`` some models insert between the tag name and its attributes.
    """
    args = {}
    for p in _PARAM_SELF_RE.finditer(body):
        args[p.group(1)] = clean_parameter(p.group(1), p.group(2))
    for p in _PARAM_PAIR_RE.finditer(body):
        if p.group(1) not in args:
            args[p.group(1)] = clean_parameter(p.group(1), p.group(2).strip())
    return args


def make_call(name: str, args: dict) -> dict:
    """Build an OpenAI-shaped ``tool_calls`` entry from a recovered name/args pair."""
    return {
        "id": f"call-proxied-{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def normalize_content(text: str):
    """Split a content string into (cleaned_text, recovered_calls)."""
    if not isinstance(text, str) or not text or "<tool_call" not in text:
        return text, []
    recovered = []
    pieces = []
    last = 0
    for tc in _TOOL_CALLS_RE.finditer(text):
        pieces.append(text[last:tc.start()])
        inner = tc.group(1) if tc.group(1) is not None else tc.group(2)
        for inv in _INVOKE_RE.finditer(inner):
            name = inv.group(1).strip()
            args = invoke_args(inv.group(2))
            if not args:
                args = {"raw": inv.group(2).strip()}
            recovered.append(make_call(name, args))
        last = tc.end()
    pieces.append(text[last:])
    cleaned = "".join(p for p in pieces if p).strip()
    if recovered:
        log.info(
            "recovered tool call(s) from content: %s",
            ", ".join(c["function"]["name"] for c in recovered),
        )
        if len(cleaned) <= 160 and re.match(
            r"^(here|the |this|a |an |run|use|below)\b", cleaned, re.IGNORECASE
        ):
            # A short wrapper ("Here is the command"), not a separate statement.
            cleaned = ""
    return cleaned, recovered


def normalize_message(msg: dict) -> int:
    """Move recovered XML tool calls out of ``content`` into ``tool_calls``."""
    if not isinstance(msg, dict):
        return 0
    recovered = []
    content = msg.get("content")
    if isinstance(content, str):
        cleaned, recovered = normalize_content(content)
        msg["content"] = cleaned
    elif isinstance(content, list):
        new_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                cleared, rec = normalize_content(part.get("text") or "")
                recovered.extend(rec)
                part["text"] = cleared
            new_parts.append(part)
        msg["content"] = new_parts
    if not recovered:
        return 0
    raw_calls = msg.get("tool_calls")
    existing: list = [c for c in raw_calls if isinstance(c, dict)] if isinstance(
        raw_calls, list
    ) else []
    existing.extend(recovered)
    msg["tool_calls"] = existing
    msg["role"] = msg.get("role") or "assistant"
    return len(recovered)
