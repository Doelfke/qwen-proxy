"""Value-level repair for tool-call arguments.

The helpers here fix individual *values* that a model degraded in transport:
prose wrappers and code fences around a command, unterminated heredocs, dropped
closing parentheses, and collapsed URL scheme separators (``https://x``). They
operate on plain strings and are shared by the XML recovery path, the
non-stream body normalizer, and the SSE stream normalizer.
"""

from __future__ import annotations

import re


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


# Collapsed scheme separator: "https:/api..." (single slash), or "https: api"-style
# degradation, but never a valid "https://" and never a trailing '<' (an XML tag such as
# </invoke>, which the repair must not touch).
_COLLAPSED_URL_RE = re.compile(r"\b(https?|wss?|file)://?([^<\s'\"\]]+)")

_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
_SHELL_KEYWORD_RE = re.compile(
    r"\b(python[23]?|node|bash|zsh|sh|cat|echo|ls|cd|grep|curl|wget)\b"
)

# The prose wrapper that sometimes leads a mangled command: a short non-shell
# sentence ("Here is the command") followed by an actual shell opener.
_PROSE_LINE_RE = re.compile(
    r"^(the|this|a |an |below|run|use|here)\b",
    re.IGNORECASE,
)
_SHELL_PREFIX_RE = re.compile(r"^(cat |cd |python3|python |bash |node |<<)")


def repair_url(value: str) -> str:
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


def repair_command(cmd: str) -> str:
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


def clean_parameter(name: str, value: str) -> str:
    """Apply the value-specific repair for a named parameter (command/url)."""
    value = _unquote(value)
    if name == "command":
        return repair_command(value)
    if name == "url":
        return repair_url(value)
    return value
