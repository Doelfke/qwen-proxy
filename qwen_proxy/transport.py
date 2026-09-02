"""Forward a client request to the upstream and relay the response back.

Handles both directions of I/O for one request: reading the client body,
building the upstream request (stripping hop-by-hop headers, forcing an
identity-encoded response), following chunked framing on the way in, and
re-serializing chat-completions responses through the normalizers before they
reach the client. Client-side aborts are absorbed here rather than surfacing as
tracebacks out of the server thread.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import urllib.error
import urllib.request

from .nonstream import normalize_response_body
from .stream import stream_normalizer

log = logging.getLogger("toolcall_proxy")

# Default timeout (seconds) for upstream requests; overridable via --timeout.
UPSTREAM_TIMEOUT = 300.0


def hop_headers(headers) -> dict:
    return {
        k: v
        for k, v in dict(headers).items()
        if k.lower()
        not in ("transfer-encoding", "connection", "content-length", "content-encoding")
    }


def is_chat_completions(path: str) -> bool:
    return "/chat/completions" in path


class QuietWriter:
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


def send_upstream_error(handler, message: str) -> None:
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


def read_upstream(resp, timeout: float | None = UPSTREAM_TIMEOUT):
    """Yield body bytes from an upstream response, following chunked framing if present."""
    te = resp.headers.get("Transfer-Encoding")
    if te:
        log.debug("upstream response Transfer-Encoding: %s", te)
    if "chunked" in (te or "").lower():
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


def send_chunked(handler, gen) -> None:
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


def proxy_request(handler, upstream: str, api_key, timeout: float | None = UPSTREAM_TIMEOUT) -> None:
    method = handler.command
    body = handler._read_body()
    log.debug("client request: %s %s", method, handler.path)
    if body:
        log.debug("client request headers:\n%s", dict(handler.headers))
        log.debug("client request body:\n%s", body.decode("utf-8", "replace"))
    headers = {
        k: v
        for k, v in handler.headers.items()
        if k.lower()
        not in (
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
            # Force an identity (uncompressed) upstream response. The proxy does not
            # decompress bodies, so a compressed reply would have its
            # Content-Encoding stripped and be forwarded still compressed -> garbage.
            "accept-encoding",
        )
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        upstream + handler.path, data=body, headers=headers, method=method
    )
    log.debug("forwarding request to %s", req.full_url)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - explicit upstream host
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        log.debug(
            "upstream error %d:\n%s",
            exc.code,
            payload.decode("utf-8", "replace"),
        )
        handler.send_response(exc.code)
        for k, v in hop_headers(exc.headers).items():
            handler.send_header(k, v)
        handler._cors()
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
        return
    except (urllib.error.URLError, OSError, socket.timeout, TimeoutError) as exc:
        send_upstream_error(handler, f"upstream unreachable: {exc}")
        return

    headers_sent = False
    try:
        is_chat = is_chat_completions(handler.path)
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
            for k, v in hop_headers(resp.headers).items():
                handler.send_header(k, v)
            handler._cors()
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            headers_sent = True
            gen = read_upstream(resp, timeout)
            if is_chat:
                gen = stream_normalizer(gen)
            send_chunked(handler, gen)
        else:
            payload = resp.read()
            log.debug(
                "upstream response %d (%s):\n%s",
                resp.status,
                resp.headers.get("Content-Type") or "",
                payload.decode("utf-8", "replace"),
            )
            if is_chat and resp.status == 200:
                raw_payload = payload
                payload = normalize_response_body(payload)
                if payload != raw_payload:
                    log.debug(
                        "normalized response sent to client:\n%s",
                        payload.decode("utf-8", "replace"),
                    )
            handler.send_response(resp.status)
            for k, v in hop_headers(resp.headers).items():
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
                send_upstream_error(handler, f"upstream read failed: {exc}")
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
