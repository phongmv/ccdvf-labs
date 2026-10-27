"""
Lab 5: Code-Driven Decisions — Pre-configured Decision Tree
===========================================================
Goal: show how to use the LLM *only as a classifier*, then let hardcoded
      Python logic own every routing decision.

Pattern:
  1. Claude receives the user's request and calls the `classify_request` tool
     to emit a structured { category, urgency, summary }.
  2. Python inspects the tool result and walks a hardcoded decision tree.
  3. The LLM never decides what action to take — only what the request *is*.

This is the "code-driven" pattern, as opposed to "LLM-driven" where you'd ask
the model to choose the next step itself.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from lib import sdk_parser

MODEL = "claude-haiku-5-5"

# ── Tool definition ───────────────────────────────────────────────────────────
# The LLM's only job: fill this schema. All decisions happen *after* in Python.

CLASSIFY_TOOL = {
    "name": "classify_request",
    "description": (
        "Classify the user's support request. "
        "Call this exactly once. Do not decide what action to take — "
        "only identify the category and urgency."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["billing", "technical", "general"],
                "description": "Primary topic of the request.",
            },
            "urgency": {
                "type": "string",
                "enum": ["urgent", "normal", "low"],
                "description": (
                    "urgent = service down / payment failed; "
                    "normal = working issue; "
                    "low = question / curiosity"
                ),
            },
            "summary": {
                "type": "string",
                "description": "One-sentence neutral summary of the request.",
            },
        },
        "required": ["category", "urgency", "summary"],
    },
}


# ── Hardcoded decision tree ───────────────────────────────────────────────────
# Every branch is explicit Python. The LLM has zero influence here.

def decide(category: str, urgency: str, summary: str) -> dict:
    """Return the action to take based on category × urgency."""

    tree = {
        ("billing",   "urgent"): {
            "action":  "ESCALATE_TO_HUMAN_AGENT",
            "message": "Payment failure detected — routing to billing specialist immediately.",
            "sla":     "< 15 min",
        },
        ("billing",   "normal"): {
            "action":  "SEND_PAYMENT_FAQ",
            "message": "Sending automated billing FAQ + self-service portal link.",
            "sla":     "< 4 h",
        },
        ("billing",   "low"): {
            "action":  "QUEUE_BILLING_REVIEW",
            "message": "Adding to billing review queue for next business day.",
            "sla":     "< 24 h",
        },
        ("technical", "urgent"): {
            "action":  "CREATE_PRIORITY_TICKET",
            "message": "Opening P1 ticket and paging on-call engineer.",
            "sla":     "< 30 min",
        },
        ("technical", "normal"): {
            "action":  "SEARCH_KNOWLEDGE_BASE",
            "message": "Searching knowledge base for matching articles.",
            "sla":     "< 2 h",
        },
        ("technical", "low"): {
            "action":  "SUGGEST_DOCUMENTATION",
            "message": "Pointing to relevant docs section.",
            "sla":     "< 48 h",
        },
        ("general",   "urgent"): {
            "action":  "ASSIGN_SUPPORT_AGENT",
            "message": "Assigning first available agent.",
            "sla":     "< 1 h",
        },
        ("general",   "normal"): {
            "action":  "SEND_AUTO_RESPONSE",
            "message": "Sending templated response + follow-up survey.",
            "sla":     "< 8 h",
        },
        ("general",   "low"): {
            "action":  "ARCHIVE_WITH_TAG",
            "message": "Tagging and archiving; no reply needed.",
            "sla":     "—",
        },
    }

    result = tree.get((category, urgency))
    if result is None:
        result = {
            "action":  "MANUAL_REVIEW",
            "message": "Unknown combination — routing to manual review.",
            "sla":     "—",
        }
    return {**result, "category": category, "urgency": urgency, "summary": summary}


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run(user_request: str) -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_request}]

    print(f"\n{'═' * 64}")
    print("  Lab 5 — Code-Driven Decision Tree")
    print(f"  Model : {MODEL}")
    print(f"{'═' * 64}")
    print(f"\n[USER]  {user_request}")

    turn = 0
    classification = None

    while True:
        turn += 1
        sdk_parser.log_turn_header(turn)
        sdk_parser.log_messages_state(messages)

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            tools=[CLASSIFY_TOOL],
            # Force the model to classify; it must not free-text a decision.
            tool_choice={"type": "any"},
            messages=messages,
        )

        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)
        sdk_parser.log_divider()

        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type != "tool_use":
                    continue
                # Classification received — no need to loop back to Claude.
                # The decision tree runs entirely in Python from here.
                classification = block.input
                sdk_parser.log_tool_execution(
                    block.name, block.input, "(classification received — handing to decision tree)"
                )

            print("\n[LOOP]  Classification done — exiting loop, running decision tree.")
            break

        if response.stop_reason == "end_turn":
            print("\n[LOOP]  stop_reason='end_turn'  →  loop complete")
            break

        print(f"\n[LOOP]  Unexpected stop_reason={response.stop_reason!r} — stopping.")
        break

    # ── Decision tree — runs in Python, not in the LLM ───────────────────────

    if classification is None:
        print("\n[DECISION]  No classification received — cannot route.")
        return

    print(f"\n{'─' * 64}")
    print("[CLASSIFICATION FROM LLM]")
    print(f"  Category : {classification['category']}")
    print(f"  Urgency  : {classification['urgency']}")
    print(f"  Summary  : {classification['summary']}")

    decision = decide(
        classification["category"],
        classification["urgency"],
        classification["summary"],
    )

    print("\n[DECISION TREE — hardcoded Python logic]")
    print(f"  Action  : {decision['action']}")
    print(f"  Next    : {decision['message']}")
    print(f"  SLA     : {decision['sla']}")
    print(f"{'─' * 64}\n")


# ── Demo ──────────────────────────────────────────────────────────────────────

DEMO_REQUESTS = [
    "My card was just charged twice for the same subscription!",
    "I can't get the API to return 200 — it keeps timing out.",
    "Do you offer educational discounts?",
    "The dashboard has been completely down for 2 hours, we're losing money.",
    "Curious how your pricing tiers work.",
]

if __name__ == "__main__":
    for req in DEMO_REQUESTS:
        run(req)
        print()
