"""OpenAI-compatible sidecar that repairs malformed tool-call output from local models.

Qwen-class models served through an OpenAI-compatible endpoint (vLLM/SGLang/...) occasionally
emit tool calls as *text* inside ``content`` instead of as structured ``tool_calls`` entries:
the native ``<tool_calls>`` XML format leaking through, plus command strings whose line breaks
were dropped (unterminated heredocs, dropped closing parentheses, stray prose wrappers,
collapsed URL scheme separators). VS Code then renders that text instead of running the tool,
and the agent loop stalls.

This proxy sits in front of the upstream server:

- upstream: ``http://127.0.0.1:8000/v1/chat/completions``
- VS Code:  ``http://127.0.0.1:8787/v1/chat/completions``

Every response body is normalized before it reaches VS Code:

1. ``<tool_calls>/<invoke>/<parameter>`` blocks found in ``content`` (streamed or not) are
   parsed and moved into the structured ``tool_calls`` array;
2. command values are repaired: prose wrappers and code fences are stripped, unterminated
   heredocs are closed, missing closing parentheses are rebalanced;
3. URL values with a collapsed scheme separator (``https://`` -> ``https:/x``) are fixed.

Stdlib only. Usage:

    python3 proxy.py --upstream http://127.0.0.1:8000 \
        --listen 127.0.0.1:8787

Then point VS Code's Custom Endpoint provider (``chatLanguageModels.json``) at
``http://127.0.0.1:8787/v1/chat/completions``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import re
import socket
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("toolcall_proxy")

# ------------------------------------------------------------------- patterns

_TOOL_CALLS_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>"
    r"|<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_INVOKE_RE = re.compile(
    r"<invoke\s+name=['\"]([^'\"]+)['\"]\s*(?:/?>)\s*(.*?)\s*(?:</invoke>)?",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    r"<parameter\s+name=['\"]([^'\"]+)['\"]\s*(?:/?>)\s*(.*?)\s*(?:</parameter>)?",
    re.DOTALL | re.IGNORECASE,
)
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
# Collapsed scheme separator: "https:/api..." (single slash), or "https: api"-style
# degradation, but never a valid "https://" and never a trailing '<' (an XML tag such as
# </invoke>, which the repair must not touch).
_COLLAPSED_URL_RE = re.compile(r"\b(https?|wss?|file)://?([^<\s'\"\]]+)")
_SHELL_KEYWORD_RE = re.compile(
    r"\b(python[23]?|node|bash|zsh|sh|cat|echo|ls|cd|grep|curl|wget)\b"
)


# ------------------------------------------------------------------- value repair


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def _repair_url(value: str) -> str:
    return _COLLAPSED_URL_RE.sub(lambda m: f"{m.group(1)}://{m.group(2)}", value)


def _strip_code_fence(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        indent = " " * (len(lines[0]) - len(lines[0].lstrip()))
        lines[0] = indent + lines[0].lstrip()[3:]
        lines = lines[1:]
    if len(lines) > 1 and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _looks_like_shell_opener(line: str) -> bool:
    return _SHELL_PREFIX_RE.match(line.lstrip()) is not None


def _has_shell_keyword(line: str) -> bool:
    return _SHELL_KEYWORD_RE.search(line) is not None


def _is_prose_line(line: str, next_line: str) -> bool:
    """True if ``line`` reads as a prose wrapper (e.g. "Here is the command:")
    immediately preceding a shell opener. Such lines are stripped before the command."""
    line = line.strip()
    return bool(
        len(line) <= 100
        and not line.startswith(("#", "-", "`", "<"))
        and "=" not in line
        and "<<" not in line
        and not _has_shell_keyword(line)
        and _PROSE_LINE_RE.match(line) is not None
        and _looks_like_shell_opener(next_line)
    )


def _repair_command(cmd: str) -> str:
    """Repair a command string whose line breaks or structure degraded in transport."""
    cmd = cmd.strip()

    # Drop a leading prose line ("Here is the command:") before a shell opener.
    if "\n" in cmd:
        first, rest = cmd.split("\n", 1)
        if _is_prose_line(first, rest):
            cmd = rest.strip()

    cmd = _strip_code_fence(cmd).strip()

    # Close an unterminated heredoc: the body is the code after the opener line.
    if _HEREDOC_RE.search(cmd):
        lines = cmd.split("\n")
        for idx, line in enumerate(lines):
            m = _HEREDOC_RE.search(line)
            if not m:
                continue
            marker = m.group(1)
            body = [ln for ln in lines[idx + 1:]]
            if any(ln.strip() == marker for ln in body):
                return cmd + "\n"
            balanced = "\n".join(body)
            opens = balanced.count("(") - balanced.count(")")
            if opens > 0:
                balanced += "\n" + ")" * opens
            return (
                "\n".join(lines[: idx + 1]) + "\n" + balanced.rstrip("\n") + "\n"
                + marker + "\n"
            )
    return cmd + "\n"


def _clean_parameter(name: str, value: str) -> str:
    value = _unquote(value)
    if name == "command":
        return _repair_command(value)
    if name == "url":
        return _repair_url(value)
    return value

# ------------------------------------------------------------------- XML recovery


def _invoke_args(body: str) -> dict:
    args = {}
    for p in _PARAM_RE.finditer(body):
        args[p.group(1)] = _clean_parameter(p.group(1), p.group(2))
    return args


def _make_call(name: str, args: dict) -> dict:
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
            args = _invoke_args(inv.group(2))
            if not args:
                args = {"raw": inv.group(2).strip()}
            recovered.append(_make_call(name, args))
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

# ------------------------------------------------------------------- non-stream body


def normalize_response_body(body: bytes) -> bytes:
    """Normalize one non-streaming chat-completions response.

    Also repairs ``tool_calls`` entries that were already structured but carry mangled
    argument values (collapsed URLs, broken command strings).
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        return body
    total = 0
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            sub = choice.get(key)
            if not isinstance(sub, dict):
                continue
            total += normalize_message(sub)
            _repair_structured_calls(sub)
    if total:
        log.info("normalized %d recovered call(s) in a non-stream response", total)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return body


def _repair_structured_calls(msg: dict) -> None:
    """Repair values inside already-structured tool_calls (URLs, commands)."""
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return
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
                new = _repair_url(args[key])
            elif key == "command":
                new = _repair_command(args[key])
            else:
                continue
            if new != args[key]:
                args[key] = new
                fixed = True
        if fixed:
            fn["arguments"] = json.dumps(args, ensure_ascii=False)

# ------------------------------------------------------------------- SSE stream repair

# Default timeout (seconds) for upstream requests; overridable via --timeout.
UPSTREAM_TIMEOUT = 300.0


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _text_event(text: str, role: bool = False) -> bytes:
    delta = {"role": "assistant", "content": text} if role else {"content": text}
    return _sse_event({"choices": [{"delta": delta, "index": 0}]})


def _tool_call_event(calls: list, role: bool = False) -> bytes:
    delta = {"role": "assistant", "tool_calls": calls} if role else {"tool_calls": calls}
    return _sse_event({"choices": [{"delta": delta, "index": 0}]})


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
# The prose wrapper that sometimes leads a mangled command: a short non-shell
# sentence ("Here is the command") followed by an actual shell opener.
_PROSE_LINE_RE = re.compile(
    r"^(the|this|a |an |below|run|use|here)\b",
    re.IGNORECASE,
)
_SHELL_PREFIX_RE = re.compile(r"^(cat |cd |python3|python |bash |node |<<)")

def _holds_unfinished_tool_block(buf: str) -> bool:
    """True if ``buf`` holds a tool block whose closing marker hasn't arrived yet.

    Either a real opener tag (``<tool_calls>``/``<invoke>``/``<parameter>``) is
    still waiting for its ``</tool_calls>``/``</tool_call>`` close, or a token is
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
    """Repair ``<tool_calls>`` XML emitted as content in a chat-completions SSE stream.

    Raw upstream bytes are first reassembled into complete SSE records
    (blank-line delimited) so a ``data:`` line is never read mid-record, even when
    a transport chunk splits one. Content deltas are then rebuffered and flushed as

    * a proper ``tool_calls`` delta once a complete
      ``</tool_calls>``/``</tool_call>`` block has arrived, with the surrounding
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
                out.append(_text_event(buf, role=not emitted))
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
            _repair_structured_calls(d)
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
                events.append(_text_event(head, role=first))
                first = False
            inner = (
                complete.group(1)
                if complete.group(1) is not None
                else complete.group(2)
            )
            calls = [
                _make_call(inv.group(1).strip(), _invoke_args(inv.group(2)))
                for inv in _INVOKE_RE.finditer(inner)
            ]
            if calls:
                log.info("recovered %d streamed tool call(s) from content", len(calls))
                events.append(_tool_call_event(calls, role=first))
                first = False
            buf = buf[complete.end() :]
            complete = _TOOL_CALLS_RE.search(buf)
        if buf:
            if not _holds_unfinished_tool_block(buf) and buf.strip():
                events.append(_text_event(buf, role=first))
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
        yield _text_event(buf, role=not emitted)


# ------------------------------------------------------------------- transport


def _hop_headers(headers) -> dict:
    return {
        k: v
        for k, v in dict(headers).items()
        if k.lower()
        not in ("transfer-encoding", "connection", "content-length", "content-encoding")
    }


def _is_chat_completions(path: str) -> bool:
    return "/chat/completions" in path


class _QuietWriter:
    """Drop-in stand-in for ``wfile`` once the client has hung up.

    ``BaseHTTPRequestHandler`` calls ``self.wfile.flush()`` after the handler
    method returns and closes the streams in ``finish()``. On an already-dead
    socket that final flush would raise ``BrokenPipeError``/``ConnectionResetError``
    (and flush-after-close raises ``ValueError``) — none of which the stdlib
    catches, so an aborted request would spam a traceback. Swapping in this
    no-op writer makes the post-handler flush and ``finish()`` harmless.
    """

    closed = False

    def write(self, _data) -> int:
        return 0

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def writable(self) -> bool:
        return False


def _send_upstream_error(handler, message: str) -> None:
    """Send a 502 JSON body reporting an upstream failure."""
    payload = json.dumps(
        {
            "error": {
                "message": message,
                "type": "proxy_error",
                "code": "upstream_unreachable",
            }
        }
    ).encode()
    handler.send_response(502)
    handler.send_header("Content-Type", "application/json")
    handler._cors()
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_upstream(resp, timeout: float | None = UPSTREAM_TIMEOUT):
    """Yield body bytes from an upstream response, following chunked framing if present."""
    if "chunked" in (resp.headers.get("Transfer-Encoding") or "").lower():
        fp = resp.fp
        if timeout is not None:
            # fp may be a BufferedReader (over the socket, or over the
            # http.client chunked reader). Only the raw IO object exposes
            # settimeout; the socket timeout set by urlopen() still applies.
            target = getattr(fp, "raw", None) or fp
            if callable(getattr(target, "settimeout", None)):
                try:
                    target.settimeout(timeout)
                except (OSError, ValueError):
                    pass
        while True:
            size_line = fp.readline(65536)
            if not size_line:
                break
            try:
                size = int(size_line.split(b";")[0].strip() or b"0", 16)
            except ValueError:
                break
            if not size:
                fp.readline(8)
                break
            buf = b""
            while len(buf) < size:
                chunk = fp.read(size - len(buf))
                if not chunk:
                    break
                buf += chunk
            if not buf:
                break
            yield buf
            fp.readline(8)
        return
    data = resp.read()
    if data:
        yield data


def _send_chunked(handler, gen) -> None:
    """Relay upstream chunks to the client; stop quietly if the client hangs up.

    A cancel in VS Code (or any abrupt client disconnect) surfaces as a
    BrokenPipe/ConnectionReset on the next write. That is a client-side event,
    not a proxy fault, so it is logged once and the connection is torn down
    here instead of propagating as a traceback out of the server thread.
    """
    for chunk in gen:
        if not isinstance(chunk, (bytes, bytearray)):
            chunk = str(chunk).encode()
        try:
            handler.wfile.write(f"{len(chunk):X}\r\n".encode() + bytes(chunk) + b"\r\n")
            handler.wfile.flush()
        except ConnectionError:
            log.info(
                "%s %s: client disconnected mid-stream; aborting forward",
                handler.command,
                handler.path,
            )
            handler._client_gone()
            return
    try:
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
    except ConnectionError:
        log.info(
            "%s %s: client disconnected before stream end; aborting forward",
            handler.command,
            handler.path,
        )
        handler._client_gone()


def _proxy_request(handler, upstream: str, api_key, timeout: float | None = UPSTREAM_TIMEOUT) -> None:
    method = handler.command
    body = handler._read_body()
    headers = {
        k: v
        for k, v in handler.headers.items()
        if k.lower() not in ("host", "content-length", "connection", "transfer-encoding")
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        upstream + handler.path, data=body, headers=headers, method=method
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - explicit upstream host
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        handler.send_response(exc.code)
        for k, v in _hop_headers(exc.headers).items():
            handler.send_header(k, v)
        handler._cors()
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
        return
    except (urllib.error.URLError, OSError, socket.timeout, TimeoutError) as exc:
        _send_upstream_error(handler, f"upstream unreachable: {exc}")
        return

    headers_sent = False
    try:
        is_chat = _is_chat_completions(handler.path)
        stream = False
        if is_chat and body:
            try:
                stream = bool(json.loads(body).get("stream"))
            except ValueError:
                pass
        if not is_chat:
            stream = (resp.headers.get("Content-Type") or "").startswith(
                "text/event-stream"
            )

        if stream:
            handler.send_response(resp.status)
            for k, v in _hop_headers(resp.headers).items():
                handler.send_header(k, v)
            handler._cors()
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            headers_sent = True
            gen = _read_upstream(resp, timeout)
            if is_chat:
                gen = stream_normalizer(gen)
            _send_chunked(handler, gen)
        else:
            payload = resp.read()
            if is_chat and resp.status == 200:
                payload = normalize_response_body(payload)
            handler.send_response(resp.status)
            for k, v in _hop_headers(resp.headers).items():
                handler.send_header(k, v)
            handler._cors()
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            headers_sent = True
            handler.wfile.write(payload)
    except ConnectionError as exc:
        # A socket on either end reset - most often the client (VS Code)
        # cancelling or aborting mid-response. There is nothing left to serve,
        # so stop cleanly instead of raising through the server thread.
        log.info(
            "%s %s: connection dropped mid-response (%s); stopping",
            handler.command,
            handler.path,
            type(exc).__name__,
        )
        handler._client_gone()
    except (http.client.HTTPException, ValueError, OSError, TimeoutError) as exc:
        # The upstream body failed while being read (truncated/malformed body,
        # reset, timeout). If nothing has been sent to the client yet, surface
        # a 502; once headers are out there is nothing left to write, so just
        # record the failure.
        if not headers_sent:
            try:
                _send_upstream_error(handler, f"upstream read failed: {exc}")
                return
            except (ConnectionError, OSError):
                pass
        else:
            log.warning(
                "%s %s: upstream error mid-stream (%s); aborting",
                handler.command,
                handler.path,
                type(exc).__name__,
            )
        handler._client_gone()
    finally:
        resp.close()

# ------------------------------------------------------------------- server


def _make_handler_class(upstream: str, api_key, timeout: float | None = UPSTREAM_TIMEOUT):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002
            log.debug(format, *args)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Headers", "Content-Type, Authorization"
            )
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _client_gone(self) -> None:
            """Tear down a client connection that hung up mid-response.

            Flagging ``close_connection`` lets ``ThreadingHTTPServer`` reclaim
            this socket after the current request, and swapping ``wfile`` for a
            no-op writer keeps ``BaseHTTPRequestHandler``'s post-handler flush
            and ``finish()`` from raising ``BrokenPipeError`` or an
            ``I/O operation on closed file`` on the dead socket.
            """
            self.close_connection = True
            self.wfile = _QuietWriter()

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("", "/health"):
                payload = json.dumps(
                    {"status": "ok", "upstream": upstream}, ensure_ascii=False
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            _proxy_request(self, upstream, api_key, timeout=timeout)

        def do_POST(self) -> None:  # noqa: N802
            _proxy_request(self, upstream, api_key, timeout=timeout)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible proxy that recovers XML tool calls from model output."
    )
    parser.add_argument("--listen", default="127.0.0.1:8787", help="address to listen on")
    parser.add_argument(
        "--upstream",
        default=os.environ.get("TOOLCALL_PROXY_UPSTREAM", "http://127.0.0.1:8000"),
        help="upstream OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TOOLCALL_PROXY_API_KEY") or None,
        help="Authorization Bearer token forwarded to the upstream",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=UPSTREAM_TIMEOUT,
        help="upstream request timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        host, port = args.listen.rsplit(":", 1)
        listen_addr = (host or "127.0.0.1", int(port))
    except ValueError:
        log.error("invalid --listen address %r (expected HOST:PORT)", args.listen)
        raise SystemExit(2)
    handler_cls = _make_handler_class(
        args.upstream.rstrip("/"), args.api_key, timeout=args.timeout
    )
    server = ThreadingHTTPServer(listen_addr, handler_cls)
    server.daemon_threads = True
    log.info(
        "listening on http://%s - upstream %s (tool-call recovery enabled)",
        args.listen,
        args.upstream.rstrip("/"),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()