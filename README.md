# qwen-proxy

OpenAI-compatible sidecar that repairs malformed tool-call output from local
models (vLLM/SGLang serving Qwen-class models).

Some models emit tool calls as *text* inside `content` — leaking
`<tool_calls>/<invoke>/<parameter>` XML or mangled command strings — instead
of structured `tool_calls`. VS Code then renders the text and the agent loop
stalls. This proxy normalizes every response before it reaches the client:

- `<tool_calls>` XML found in `content` (streamed or not) is parsed and moved
  into the structured `tool_calls` array.
- Command values are repaired: prose wrappers and code fences stripped,
  unterminated heredocs closed, missing closing parentheses rebalanced.
- URL values with a collapsed scheme separator (`https:/x`) are fixed.

Stdlib only — no dependencies.

## Usage

```sh
python3 proxy.py --upstream http://127.0.0.1:8000 --listen 127.0.0.1:8787
```

Then point the client at `http://127.0.0.1:8787/v1/chat/completions`.

## Options

| Flag         | Default | Description                              |
| ------------ | ------- | ---------------------------------------- |
| `--listen`   | `127.0.0.1:8787` | address to listen on              |
| `--upstream` | `127.0.0.1:8000` | upstream OpenAI-compatible base URL  |
| `--api-key`  | `TOOLCALL_PROXY_API_KEY` env | bearer token forwarded upstream   |
| `--timeout`  | `300` | upstream timeout in seconds              |
