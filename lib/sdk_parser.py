"""
sdk_parser.py — Logging helpers that format Claude API response fields
for educational display: stop_reason, content blocks, and message history.
"""

STOP_REASON_DESCRIPTIONS = {
    "end_turn":       "Claude finished naturally → loop is complete",
    "tool_use":       "Claude wants a tool result → execute and loop again",
    "max_tokens":     "Hit the max_tokens ceiling → raise it or stream",
    "stop_sequence":  "Hit a custom stop sequence",
    "pause_turn":     "Server-tool mid-turn pause → re-send to continue",
    "refusal":        "Safety decline → check stop_details.category",
}

_WIDE  = "═" * 62
_THIN  = "─" * 62


def log_turn_header(turn: int) -> None:
    print(f"\n{_WIDE}")
    print(f"  TURN {turn}")
    print(_WIDE)


def log_stop_reason(response) -> None:
    reason = response.stop_reason
    desc   = STOP_REASON_DESCRIPTIONS.get(reason, "unknown")
    print(f"\n[STOP_REASON]  {reason!r:<16}  →  {desc}")


def log_content_blocks(response) -> None:
    blocks = response.content
    print(f"\n[CONTENT]  {len(blocks)} block(s)")
    for i, block in enumerate(blocks):
        btype = block.type
        if btype == "text":
            preview = block.text[:80]
            ellipsis = "…" if len(block.text) > 80 else ""
            print(f"  [{i}] text      : {preview!r}{ellipsis}")
        elif btype == "tool_use":
            print(f"  [{i}] tool_use  : name={block.name!r}  id={block.id}")
            print(f"       input     : {block.input}")
        elif btype == "thinking":
            print(f"  [{i}] thinking  : ({len(block.thinking)} chars, summarized/omitted)")
        else:
            print(f"  [{i}] {btype:<10}: ...")


def log_tool_execution(name: str, tool_input: dict, result: str) -> None:
    print(f"\n[TOOL EXEC]  {name}({tool_input})")
    print(f"             → {result!r}")


def log_messages_state(messages: list) -> None:
    print(f"\n[MESSAGES]  {len(messages)} item(s) in history:")
    for i, msg in enumerate(messages):
        role    = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:60]
            ellipsis = "…" if len(content) > 60 else ""
            print(f"  [{i}] {role:<12} str   : {preview!r}{ellipsis}")
        elif isinstance(content, list):
            types = []
            for b in content:
                t = b.get("type") if isinstance(b, dict) else getattr(b, "type", "?")
                types.append(t)
            print(f"  [{i}] {role:<12} list  : [{', '.join(types)}]")
        else:
            print(f"  [{i}] {role:<12} mixed : {type(content).__name__}")


def log_append(role: str, description: str) -> None:
    print(f"\n[APPEND ▶]  messages ← {role!r}  ({description})")


def log_divider() -> None:
    print(f"\n{_THIN}")
