"""Repair a streamed (SSE) chat-completions response.

Wraps the upstream chunk iterator and re-emits SSE records with any tool-call
XML pulled out of ``content`` into structured ``tool_calls``, while holding back
text that could still turn into a tool block so a partial tag split across chunks
is never emitted as text in pieces.
"""

from __future__ import annotations

import json
import logging
import re

from .nonstream import repair_structured_calls
from .recovery import _INVOKE_RE, _TOOL_CALLS_RE, invoke_args, make_call

log = logging.getLogger("toolcall_proxy")


def sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def text_event(text: str, role: bool = False) -> bytes:
    delta = {"role": "assistant", "content": text} if role else {"content": text}
    return sse_event({"choices": [{"delta": delta, "index": 0}]})


def tool_call_event(calls: list, role: bool = False) -> bytes:
    delta = {"role": "assistant", "tool_calls": calls} if role else {"tool_calls": calls}
    return sse_event({"choices": [{"delta": delta, "index": 0}]})


_KNOWN_TOOL_TAGS = (
    "<tool_calls",
    "</tool_calls",
    "<tool_call",
    "</tool_call",
    "<invoke",
    "</invoke",
    "<parameter",
    "</parameter",
)

# Compiled once (module scope) rather than re-compiled on every call.
_SSE_RECORD_RE = re.compile(rb"\r?\n\r?\n")


def _holds_unfinished_tool_block(buf: str) -> bool:
    """True if ``buf`` holds a tool block whose closing marker hasn't arrived yet.

    Either a real opener tag (````/``< /tool_call>`` close, or a token is
    cut off mid-tag (``...<tool_ca``) and the next chunks will complete it.
    """
    low = buf.lower()
    if any(tag in low for tag in ("<tool_calls", "<invoke", "<parameter")):
        return True
    idx = low.rfind("<")
    if idx < 0:
        return False
    frag = low[idx:]
    return "<" not in frag[1:] and any(
        t.startswith(frag) and t != frag for t in _KNOWN_TOOL_TAGS
    )


def stream_normalizer(gen):
    """Repair ````/``< /tool_call>`` block has arrived, with the surrounding
      text flushed verbatim around it;
    * text that could still become a tool block (a tag cut off mid-token, or an
      open tag still waiting for its close) is *held back* until the next record
      arrives, so a partial ``<tool_...`` split across chunks is never emitted as
      text in pieces;
    * all other text is flushed verbatim, so a non-XML ``<tool_call`` prefix is
      never swallowed;
    * metadata-only records (``role``, ``finish_reason``, usage) and non-data
      records are passed through unchanged.
    """
    buf = ""
    raw = b""
    emitted = False

    def process_record(record: bytes) -> list:
        """Normalize one complete SSE record; returns the bytes to emit instead."""
        nonlocal buf, emitted
        line = record.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            return [record + b"\n\n"]
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            out: list = []
            if buf:
                out.append(text_event(buf, role=not emitted))
                buf = ""
                emitted = True
            out.append(b"data: [DONE]\n\n")
            return out
        try:
            evt = json.loads(data)
        except ValueError:
            return [record + b"\n\n"]
        choices = evt.get("choices") if isinstance(evt, dict) else None
        if not isinstance(choices, list):
            return [record + b"\n\n"]
        consumed = False
        for c in choices:
            if not isinstance(c, dict):
                continue
            d = c.get("delta")
            if not isinstance(d, dict):
                continue
            content = d.get("content")
            if isinstance(content, str) and content:
                d.pop("content")
                buf += content
                consumed = True
            repair_structured_calls(d)
        if not consumed:
            # Metadata-only record (role, finish_reason, usage) - pass through.
            # A role was already delivered (or none needed), so don't re-emit it.
            emitted = True
            return [record + b"\n\n"]
        events = []
        first = not emitted
        complete = _TOOL_CALLS_RE.search(buf)
        while complete:
            head = buf[: complete.start()]
            if head:
                events.append(text_event(head, role=first))
                first = False
            inner = (
                complete.group(1)
                if complete.group(1) is not None
                else complete.group(2)
            )
            calls = [
                make_call(
                    inv.group(1).strip(),
                    invoke_args(inv.group(2)),
                )
                if invoke_args(inv.group(2))
                else make_call(inv.group(1).strip(), {"raw": inv.group(2).strip()})
                for inv in _INVOKE_RE.finditer(inner)
            ]
            if calls:
                log.info("recovered %d streamed tool call(s) from content", len(calls))
                events.append(tool_call_event(calls, role=first))
                first = False
            buf = buf[complete.end() :]
            complete = _TOOL_CALLS_RE.search(buf)
        if buf:
            if not _holds_unfinished_tool_block(buf) and buf.strip():
                events.append(text_event(buf, role=first))
                buf = ""
        if events:
            emitted = True
        return events

    for item in gen:
        if not isinstance(item, (bytes, bytearray)):
            continue
        raw += bytes(item)
        m = _SSE_RECORD_RE.search(raw)
        while m:
            record, raw = raw[: m.start()], raw[m.end() :]
            for out in process_record(record):
                yield out
            m = _SSE_RECORD_RE.search(raw)
    if raw.strip():
        # A trailing record written without the usual blank line before EOF.
        for out in process_record(raw):
            yield out
    if buf:
        log.debug("flushing held stream content at end of stream: %r", buf)
        yield text_event(buf, role=not emitted)
