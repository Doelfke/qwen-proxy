"""qwen_proxy: an OpenAI-compatible sidecar that repairs malformed tool-call output.

Qwen-class models served through an OpenAI-compatible endpoint (vLLM/SGLang/...)
occasionally emit tool calls as *text* inside ``content`` instead of as
structured ``tool_calls`` entries: the native ``<tool_calls>`` XML format leaking
through, plus command strings whose line breaks were dropped (unterminated
heredocs, dropped closing parentheses, stray prose wrappers, collapsed URL
scheme separators). VS Code then renders that text instead of running the tool,
and the agent loop stalls.

This package sits in front of the upstream server and normalizes every response
body before it reaches the client:

1. ``<tool_calls>/<invoke>/<parameter>`` blocks found in ``content`` (streamed or
   not) are parsed and moved into the structured ``tool_calls`` array;
2. command values are repaired: prose wrappers and code fences are stripped,
   unterminated heredocs are closed, missing closing parentheses are rebalanced;
3. URL values with a collapsed scheme separator (``https://`` -> ``https:/x``)
   are fixed.

Stdlib only. Layout:

- :mod:`qwen_proxy.repair`     — value-level repairs (commands, URLs).
- :mod:`qwen_proxy.recovery`   — recover XML tool calls out of ``content``.
- :mod:`qwen_proxy.nonstream`  — normalize a non-streaming response body.
- :mod:`qwen_proxy.stream`     — repair a streamed (SSE) response.
- :mod:`qwen_proxy.transport`  — forward one request to the upstream + relay back.
- :mod:`qwen_proxy.server`     — the HTTP server / request handler.

Run with ``python3 proxy.py`` (see the module docstring of ``proxy.py``).
"""

from __future__ import annotations

import argparse
import logging
import os
from http.server import ThreadingHTTPServer

from .server import make_handler_class
from .transport import UPSTREAM_TIMEOUT

log = logging.getLogger("toolcall_proxy")


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
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="log verbosity (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="write the full log to PATH; the console is then limited to WARNING+",
    )
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    level = getattr(logging, args.log_level)
    logging.basicConfig(level=level, format=fmt)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        log.addHandler(file_handler)
        log.setLevel(min(level, logging.DEBUG))
        # Keep console output quiet while the full log goes to the file.
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.WARNING)
    try:
        host, port = args.listen.rsplit(":", 1)
        listen_addr = (host or "127.0.0.1", int(port))
    except ValueError:
        log.error("invalid --listen address %r (expected HOST:PORT)", args.listen)
        raise SystemExit(2)
    handler_cls = make_handler_class(
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
