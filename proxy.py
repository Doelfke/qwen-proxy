#!/usr/bin/env python3
"""Entry point for the qwen-proxy sidecar.

Kept as a thin wrapper so the documented command keeps working unchanged::

    python3 proxy.py --upstream http://127.0.0.1:8000 \
        --listen 127.0.0.1:8787

The implementation lives in the :mod:`qwen_proxy` package (see its module
docstring for the layout). This file only wires up ``main()``.

Debug logging (raw request/response bodies and recovery events; streamed tokens
are not logged one record at a time)::

    python3 proxy.py --upstream http://127.0.0.1:8000 \
        --log-level DEBUG --log-file /tmp/proxy.log

Then point VS Code's Custom Endpoint provider (``chatLanguageModels.json``) at
``http://127.0.0.1:8787/v1/chat/completions``.
"""

from __future__ import annotations

import os
import sys

# Make the package importable when run directly from the repo root, regardless
# of how it is invoked (``python3 proxy.py`` vs. ``python3 -m qwen_proxy``).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_proxy import main  # noqa: E402


if __name__ == "__main__":
    main()
