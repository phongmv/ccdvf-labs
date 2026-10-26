"""
Lab 4: stop_reason in the Claude API — Weather Tool Use
========================================================
Goal: show how stop_reason drives a manual agentic loop.

  Turn 1:  Claude returns stop_reason="tool_use"
           → we execute the weather tool(s) and append results
  Turn 2:  Claude returns stop_reason="end_turn"
           → loop is complete, print the final answer

Uses lib/sdk_parser to format every SDK field for learning.
"""

import sys
import os

# Make lib/ importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from lib import sdk_parser

MODEL = "claude-haiku-5-5"

# ── Tool definition ──────────────────────────────────────────────────────────

WEATHER_TOOL = {
    "name": "get_weather",
    "description": (
        "Return current weather conditions for a given city. "
        "Call this once per city the user asks about."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Hanoi' or 'London'",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit. Defaults to celsius.",
            },
        },
        "required": ["city"],
    },
}


# ── Mock implementation ───────────────────────────────────────────────────────

_MOCK_WEATHER: dict[str, dict[str, str]] = {
    "hanoi":    {"celsius": "32°C, humid, afternoon showers", "fahrenheit": "90°F, humid, afternoon showers"},
    "london":   {"celsius": "14°C, overcast, light drizzle",  "fahrenheit": "57°F, overcast, light drizzle"},
    "tokyo":    {"celsius": "22°C, clear skies",              "fahrenheit": "72°F, clear skies"},
    "new york": {"celsius": "19°C, partly cloudy",            "fahrenheit": "66°F, partly cloudy"},
    "paris":    {"celsius": "17°C, sunny intervals",          "fahrenheit": "63°F, sunny intervals"},
}


def get_weather(city: str, unit: str = "celsius") -> str:
    unit = unit or "celsius"
    data = _MOCK_WEATHER.get(city.lower())
    if data:
        return data.get(unit, data["celsius"])
    return f"Weather data unavailable for {city}"


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run(user_prompt: str) -> None:
    client = anthropic.Anthropic()

    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    print(f"\n{'═' * 62}")
    print("  Lab 4 — stop_reason: tool_use  →  end_turn")
    print(f"  Model : {MODEL}")
    print(f"{'═' * 62}")
    print(f"\n[USER] {user_prompt}")

    turn = 0
    while True:
        turn += 1
        sdk_parser.log_turn_header(turn)
        sdk_parser.log_messages_state(messages)

        # ── API call ────────────────────────────────────────────────────────
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=[WEATHER_TOOL],
            messages=messages,
        )

        # ── Always inspect stop_reason first ────────────────────────────────
        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)
        sdk_parser.log_divider()

        # ── Branch on stop_reason ────────────────────────────────────────────

        if response.stop_reason == "end_turn":
            # Claude is done — no tool_use blocks, just a text answer.
            print("\n[LOOP]  stop_reason='end_turn'  →  loop complete, printing answer")
            final_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            print(f"\n{'─' * 62}")
            print("[FINAL ANSWER]")
            print(final_text)
            print(f"{'─' * 62}")
            break

        if response.stop_reason == "tool_use":
            # ── Step 1: append the assistant turn (with tool_use blocks) ────
            messages.append({"role": "assistant", "content": response.content})
            sdk_parser.log_append(
                "assistant",
                "contains tool_use block(s) — must be kept so Claude sees its own call",
            )

            # ── Step 2: execute each requested tool ─────────────────────────
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = get_weather(**block.input)
                sdk_parser.log_tool_execution(block.name, block.input, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # must match the tool_use block's id
                    "content": result,
                })

            # ── Step 3: append tool results as a new user turn ───────────────
            messages.append({"role": "user", "content": tool_results})
            sdk_parser.log_append(
                "user",
                f"{len(tool_results)} tool_result block(s) — Claude reads these next turn",
            )
            continue

        # Unexpected stop reason — stop safely.
        print(f"\n[LOOP]  Unexpected stop_reason={response.stop_reason!r} — stopping.")
        break


if __name__ == "__main__":
    run("What's the weather right now in Hanoi and Tokyo?")
