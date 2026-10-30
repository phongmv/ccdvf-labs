"""
Hub and Spoke — Multi-Agent Architecture
=========================================
Architecture:

       User Query
           │
           ▼
    ┌─────────────┐
    │  CentralHub │  ← the ONLY entry point; owns all routing & state
    └──────┬──────┘
           │  Claude decides which spoke(s) to call
     ┌─────┴──────┐
     ▼            ▼
 WeatherSpoke  CalculatorSpoke
     │            │
     └─────┬──────┘
           │  results flow back to Hub
           ▼
    ┌─────────────┐
    │  CentralHub │  aggregates results → final answer
    └─────────────┘

STRICT COMMUNICATION RULE:
  - Spokes NEVER know about each other.
  - Spokes NEVER call the Hub.
  - All data flows through the Hub exclusively.
"""

import os
import sys
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from lib import sdk_parser

MODEL       = "claude-haiku-5-5"
MAX_TURNS   = 10

# ═══════════════════════════════════════════════════════════════════════════════
# SPOKE AGENTS
# Each spoke is a self-contained class with ONE method: handle(query) → str.
# Spokes have no imports of other agents and no reference to the Hub.
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherSpokeAgent:
    """
    Spoke #1 — WeatherSpokeAgent
    Handles ONLY weather-related queries.
    Returns weather data as a plain string.
    Has no knowledge of CalculatorSpokeAgent or CentralHub.
    """

    _MOCK_DB: ClassVar[dict[str, dict]] = {
        "hanoi":     {"temp_c": 32, "condition": "humid, afternoon showers",    "humidity": "88%"},
        "london":    {"temp_c": 14, "condition": "overcast, light drizzle",      "humidity": "75%"},
        "tokyo":     {"temp_c": 22, "condition": "clear skies",                  "humidity": "55%"},
        "new york":  {"temp_c": 19, "condition": "partly cloudy",                "humidity": "62%"},
        "paris":     {"temp_c": 17, "condition": "sunny intervals",              "humidity": "60%"},
        "singapore": {"temp_c": 30, "condition": "tropical, humid",              "humidity": "90%"},
    }

    def handle(self, city: str, unit: str = "celsius") -> str:
        """
        Called ONLY by CentralHub — never by another Spoke.
        Returns a formatted weather string.
        """
        data = self._MOCK_DB.get(city.lower().strip())
        if not data:
            return f"[WeatherSpoke] No weather data for '{city}'. Known cities: {list(self._MOCK_DB)}"

        temp = data["temp_c"] if unit == "celsius" else round(data["temp_c"] * 9/5 + 32)
        unit_label = "°C" if unit == "celsius" else "°F"
        return (
            f"[WeatherSpoke] {city.title()}: "
            f"{temp}{unit_label}, {data['condition']}, "
            f"humidity {data['humidity']}"
        )


class CalculatorSpokeAgent:
    """
    Spoke #2 — CalculatorSpokeAgent
    Handles ONLY arithmetic expressions.
    Returns the computed result as a plain string.
    Has no knowledge of WeatherSpokeAgent or CentralHub.
    """

    # Safe subset of math operations
    _ALLOWED_NAMES: ClassVar[dict] = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
    }

    def handle(self, expression: str) -> str:
        """
        Called ONLY by CentralHub — never by another Spoke.
        Evaluates a safe math expression and returns the result.
        """
        try:
            # Restrict eval to arithmetic + safe builtins
            result = eval(expression, {"__builtins__": {}}, self._ALLOWED_NAMES)
            return f"[CalculatorSpoke] {expression} = {result}"
        except ZeroDivisionError:
            return f"[CalculatorSpoke] Error: division by zero in '{expression}'"
        except (ValueError, SyntaxError, TypeError, NameError) as exc:
            return f"[CalculatorSpoke] Error evaluating '{expression}': {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# CENTRAL HUB
# Owns all Spoke instances. Uses Claude to decide routing.
# All requests come in here; all results are aggregated here.
# ═══════════════════════════════════════════════════════════════════════════════

class CentralHub:
    """
    CentralHub — the coordinator.

    Responsibilities:
      1. Accept user queries.
      2. Ask Claude which spoke(s) to call (tool_use routing).
      3. Dispatch to the right Spoke instance.
      4. Collect and aggregate results.
      5. Return the final answer to the caller.

    The Hub is the ONLY class that holds references to Spoke objects.
    """

    # ── System prompt: Hub role ──────────────────────────────────────────────
    _SYSTEM: ClassVar[str] = """\
You are a central coordinator in a Hub-and-Spoke multi-agent system.

You have two specialized Spoke agents available as tools:
  • weather_spoke  — call this for any weather-related question.
  • calculator_spoke — call this for any arithmetic/math calculation.

Rules:
  1. Analyse the user query and call the appropriate spoke tool(s).
  2. You may call both tools in parallel if the query needs both.
  3. After receiving the spoke results, synthesise a clear, final answer.
  4. Never answer a weather or math question from your own knowledge — always
     use the appropriate spoke tool.
"""

    # ── Tool definitions that represent Spokes ───────────────────────────────
    _SPOKE_TOOLS: ClassVar[list] = [
        {
            "name": "weather_spoke",
            "description": (
                "Get current weather for a city. "
                "Call this for any weather-related question."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city, e.g. 'Hanoi', 'London'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default: celsius.",
                    },
                },
                "required": ["city"],
            },
        },
        {
            "name": "calculator_spoke",
            "description": (
                "Evaluate an arithmetic expression and return the result. "
                "Call this for any math or calculation question."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A safe Python arithmetic expression, e.g. '2 + 2', "
                            "'round(3.14159 * 10**2, 2)', '(100 - 32) * 5/9'."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    ]

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

        # ── Spoke instances: ONLY the Hub holds these references ─────────────
        self._spokes: dict[str, object] = {
            "weather_spoke":    WeatherSpokeAgent(),
            "calculator_spoke": CalculatorSpokeAgent(),
        }

    # ── Internal: dispatch a single tool call to the right Spoke ─────────────

    def _dispatch_to_spoke(self, tool_name: str, tool_input: dict) -> str:
        """
        Route one tool-use request to the correct Spoke.
        This is the ONLY place where Spoke.handle() is called.
        """
        spoke = self._spokes.get(tool_name)
        if spoke is None:
            return f"[Hub] Unknown spoke: {tool_name!r}"

        print(f"\n  ┌─ [HUB → SPOKE]  dispatching to {tool_name.upper()}")
        print(f"  │   input   : {tool_input}")

        result = spoke.handle(**tool_input)

        print(f"  │   result  : {result}")
        print("  └─ [SPOKE → HUB]  result returned to Hub (NOT to any other Spoke)")
        return result

    # ── Public entry point ────────────────────────────────────────────────────

    def process(self, user_query: str) -> str:
        """
        Accept a user query, route it through Claude + Spokes, return final answer.
        """
        messages: list[dict] = [{"role": "user", "content": user_query}]

        _banner("HUB AND SPOKE — Multi-Agent Routing")
        print(f"  Model         : {MODEL}")
        print(f"  Registered spokes: {list(self._spokes)}")
        print(f"\n[USER → HUB]  {user_query!r}")

        for turn in range(1, MAX_TURNS + 1):
            sdk_parser.log_turn_header(turn)
            sdk_parser.log_messages_state(messages)

            # ── Hub calls Claude to decide routing ────────────────────────────
            print("\n[HUB → CLAUDE]  asking Claude which spoke(s) to call …")
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=self._SYSTEM,
                tools=self._SPOKE_TOOLS,
                messages=messages,
            )

            sdk_parser.log_stop_reason(response)
            sdk_parser.log_content_blocks(response)
            sdk_parser.log_divider()

            # ── Done: Claude composed the final answer ────────────────────────
            if response.stop_reason == "end_turn":
                final = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                print("\n[CLAUDE → HUB]  no more spokes needed — composing final answer")
                print("\n" + "─" * 62)
                print("[FINAL ANSWER TO USER]")
                print(final)
                print("─" * 62)
                return final

            # ── Routing: Claude chose one or more Spokes ─────────────────────
            if response.stop_reason == "tool_use":
                # Step 1: save the assistant turn (contains tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})
                sdk_parser.log_append(
                    "assistant",
                    "Claude's routing decision (tool_use blocks) saved to history",
                )

                # Step 2: execute EACH tool call by dispatching through the Hub
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    print(
                        f"\n[HUB]  Claude chose spoke={block.name!r}  "
                        f"(strict rule: Hub dispatches, Spokes never call each other)"
                    )
                    result = self._dispatch_to_spoke(block.name, block.input)

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,   # must match the tool_use block id
                        "content":     result,
                    })

                # Step 3: return all spoke results to Claude via Hub
                messages.append({"role": "user", "content": tool_results})
                sdk_parser.log_append(
                    "user",
                    f"{len(tool_results)} spoke result(s) delivered to Claude via Hub",
                )
                continue

            print(f"\n[HUB]  Unexpected stop_reason={response.stop_reason!r} — stopping.")
            break

        print(f"\n[HUB]  MAX_TURNS ({MAX_TURNS}) reached.")
        return "[Hub] Could not complete the request within the turn limit."


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO — shows three routing scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    hub = CentralHub()

    queries = [
        # Scenario 1: only WeatherSpoke needed
        "What's the weather in Tokyo right now?",

        # Scenario 2: only CalculatorSpoke needed
        "What is 1234 * 5678?",

        # Scenario 3: BOTH spokes needed in one query
        "What is the temperature in Hanoi in Fahrenheit, and what is 32 + 65 * 2?",
    ]

    for i, query in enumerate(queries, 1):
        _banner(f"DEMO SCENARIO {i}")
        hub.process(query)
        print()


if __name__ == "__main__":
    main()
