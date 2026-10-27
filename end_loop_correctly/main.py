"""
End Loop Correctly — Max Iterations Guard
==========================================
Goal: show how to add a hard cap to a while-loop agentic loop using an
      external variable (MAX_ITERATIONS) so the loop can never run forever.

Pattern:
  - The natural exit is stop_reason == "end_turn"
  - MAX_ITERATIONS is the safety exit — tune it without touching loop logic
  - Each iteration increments a counter checked at the top of the loop
  - When the cap is hit, the loop exits cleanly and reports partial work
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from lib import sdk_parser

MODEL = "claude-haiku-5-5"

# ── External variable — tune this without touching the loop ───────────────────
MAX_ITERATIONS = 10

# ── Tool definition ───────────────────────────────────────────────────────────

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": (
        "Evaluate a simple arithmetic expression and return the numeric result. "
        "Call this for each individual calculation step."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python-evaluable arithmetic expression, e.g. '12 * 4 + 7'.",
            },
            "step_label": {
                "type": "string",
                "description": "Short label for this step, e.g. 'subtotal', 'tax', 'total'.",
            },
        },
        "required": ["expression", "step_label"],
    },
}


# ── Mock calculator ───────────────────────────────────────────────────────────

def calculate(expression: str, step_label: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{step_label} = {result}"
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, NameError) as e:
        return f"Error evaluating '{expression}': {e}"


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run(user_prompt: str) -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_prompt}]

    print(f"\n{'═' * 64}")
    print("  End Loop Correctly — Max Iterations Guard")
    print(f"  Model         : {MODEL}")
    print(f"  MAX_ITERATIONS: {MAX_ITERATIONS}")
    print(f"{'═' * 64}")
    print(f"\n[USER]  {user_prompt}")

    iteration = 0                      # counter lives outside the loop
    exit_reason = "unknown"

    while iteration < MAX_ITERATIONS:  # guard checked before every turn
        iteration += 1
        sdk_parser.log_turn_header(iteration)
        print(f"  [iteration {iteration}/{MAX_ITERATIONS}]")
        sdk_parser.log_messages_state(messages)

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=[CALCULATOR_TOOL],
            messages=messages,
        )

        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)
        sdk_parser.log_divider()

        # ── Natural exit ─────────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            exit_reason = "end_turn"
            final = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            print(f"\n[LOOP]  Natural exit at iteration {iteration}/{MAX_ITERATIONS}")
            print(f"\n{'─' * 64}")
            print("[FINAL ANSWER]")
            print(final)
            print(f"{'─' * 64}")
            break

        # ── Tool use — execute and loop ──────────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            sdk_parser.log_append("assistant", "tool_use block(s) appended")

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = calculate(**block.input)
                sdk_parser.log_tool_execution(block.name, block.input, result)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})
            sdk_parser.log_append(
                "user", f"{len(tool_results)} tool_result(s) appended"
            )
            continue

        # ── Unexpected stop reason ───────────────────────────────────────────
        exit_reason = f"unexpected:{response.stop_reason}"
        print(f"\n[LOOP]  Unexpected stop_reason={response.stop_reason!r} — stopping.")
        break

    else:
        # ── Safety exit — while condition became False ────────────────────────
        exit_reason = "max_iterations"
        print(
            f"\n[LOOP]  MAX_ITERATIONS ({MAX_ITERATIONS}) reached — "
            "stopping to prevent infinite loop."
        )
        print("        Increase MAX_ITERATIONS or simplify the task if more steps are needed.")

    print(f"\n[EXIT]  reason={exit_reason!r}  iterations_used={iteration}/{MAX_ITERATIONS}\n")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        "A store sells 3 items: apples at $1.20 each, bananas at $0.45 each, "
        "and oranges at $2.10 each. A customer buys 5 apples, 8 bananas, and "
        "3 oranges. Calculate the subtotal for each item, then the order total, "
        "then apply a 10% discount, then add 8% sales tax on the discounted price."
    )
