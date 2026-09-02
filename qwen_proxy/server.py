"""The HTTP server and its request handler.

Builds a ``BaseHTTPRequestHandler`` subclass bound to the configured upstream,
and runs it on a :class:`ThreadingHTTPServer`. The handler adds CORS headers,
exposes ``/health``, and forwards everything else through
:func:`qwen_proxy.transport.proxy_request`.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler

from .transport import QuietWriter, proxy_request

log = logging.getLogger("toolcall_proxy")


def make_handler_class(upstream: str, api_key, timeout: float | None):
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
            # Deliberate monkey-patch: http.server flushes wfile after do_* returns
            # even when the socket is dead. Pylance types wfile as BufferedIOBase.
            self.wfile = QuietWriter()  # type: ignore[assignment]

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
            proxy_request(self, upstream, api_key, timeout=timeout)

        def do_POST(self) -> None:  # noqa: N802
            proxy_request(self, upstream, api_key, timeout=timeout)

    return Handler
