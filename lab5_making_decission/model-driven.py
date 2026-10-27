"""
Lab 5: Model-Driven Decisions
==============================
Contrast with code-driven.py:

  code-driven  →  LLM classifies only; Python owns the decision tree
  model-driven →  LLM receives the actions as tools and picks one itself;
                  Python just executes whatever the model chose

The system prompt below contains the block labelled
  "# ── DECISION RULES (used to go here) ──"
This is where teams historically hardcoded routing logic in the prompt.
Modern Claude reasons through this without explicit rules, but the block
is preserved for demonstration — it shows exactly what was being replaced.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from lib import sdk_parser

MODEL = "claude-haiku-5-5"

# ── System prompt ─────────────────────────────────────────────────────────────
# The "DECISION RULES" section is shown for contrast with the code-driven
# approach.  Claude does not need these rules — it infers the right action
# from context — but they illustrate what used to live here.

SYSTEM_PROMPT = """\
You are a customer support triage agent.

Your job: read the user's request and call exactly one action tool.
Do not reply in plain text. Always call a tool.

# ── DECISION RULES (used to go here) ────────────────────────────────────────
# In earlier prompt-driven systems, teams wrote explicit IF/ELSE logic here:
#
#   IF topic == "billing" AND urgency == "urgent"  → call escalate_to_human
#   IF topic == "billing" AND urgency == "normal"  → call send_faq
#   IF topic == "technical" AND urgency == "urgent" → call create_ticket(P1)
#   IF topic == "technical" AND urgency == "normal" → call create_ticket(P3)
#   IF topic == "general"                           → call archive_request
#
# Modern Claude reasons through this without the explicit rules.
# The block above is kept only for learning purposes — it shows exactly
# what was being replaced when teams moved to model-driven decisions.
# ─────────────────────────────────────────────────────────────────────────────
"""

# ── Action tools — the model picks one ───────────────────────────────────────
# In code-driven.py the model only filled a classifier schema.
# Here the action IS the tool — the model's tool choice IS the decision.

ACTION_TOOLS = [
    {
        "name": "escalate_to_human",
        "description": (
            "Route the request to a human agent immediately. "
            "Use for urgent billing failures, security incidents, or situations "
            "that cannot be resolved by automation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this needs a human — one sentence.",
                },
                "urgency_note": {
                    "type": "string",
                    "description": "What the agent should know before picking up.",
                },
            },
            "required": ["reason", "urgency_note"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Open a support ticket for technical or billing issues "
            "that do not require immediate human intervention."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3"],
                    "description": "P1=service down, P2=degraded, P3=question.",
                },
                "summary": {
                    "type": "string",
                    "description": "Issue summary for the ticket title.",
                },
            },
            "required": ["priority", "summary"],
        },
    },
    {
        "name": "send_knowledge_base_article",
        "description": (
            "Send the user a link to a relevant help article. "
            "Use for non-urgent technical how-to questions or billing FAQs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic keyword to look up, e.g. 'invoice', 'rate-limit'.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this article matches the request.",
                },
            },
            "required": ["topic", "reason"],
        },
    },
    {
        "name": "archive_request",
        "description": (
            "Tag and archive the request with no further action. "
            "Use for general curiosity questions or feedback that needs no reply."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Category tag, e.g. 'pricing-inquiry', 'product-feedback'.",
                },
                "summary": {
                    "type": "string",
                    "description": "One-sentence summary for the archive record.",
                },
            },
            "required": ["tag", "summary"],
        },
    },
]

# ── Mock action executors ─────────────────────────────────────────────────────

def escalate_to_human(reason: str, urgency_note: str) -> str:
    return f"✓ Paged on-call agent. Reason: {reason} | Note: {urgency_note}"

def create_support_ticket(priority: str, summary: str) -> str:
    ticket_id = f"TKT-{abs(hash(summary)) % 10000:04d}"
    return f"✓ Ticket {ticket_id} created at priority {priority}: {summary}"

def send_knowledge_base_article(topic: str, reason: str) -> str:
    return f"✓ Sent KB article for topic='{topic}'. Reason: {reason}"

def archive_request(tag: str, summary: str) -> str:
    return f"✓ Archived with tag='{tag}': {summary}"

EXECUTORS = {
    "escalate_to_human":           escalate_to_human,
    "create_support_ticket":       create_support_ticket,
    "send_knowledge_base_article": send_knowledge_base_article,
    "archive_request":             archive_request,
}


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run(user_request: str) -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_request}]

    print(f"\n{'═' * 64}")
    print("  Lab 5 — Model-Driven Decision")
    print(f"  Model : {MODEL}")
    print(f"{'═' * 64}")
    print(f"\n[USER]  {user_request}")

    turn = 0
    while True:
        turn += 1
        sdk_parser.log_turn_header(turn)
        sdk_parser.log_messages_state(messages)

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            tools=ACTION_TOOLS,
            # "auto" — the model freely chooses which action tool to call.
            # This is the key difference from code-driven: no classification
            # step, no Python routing — the model picks the action directly.
            tool_choice={"type": "auto"},
            messages=messages,
        )

        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)
        sdk_parser.log_divider()

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            sdk_parser.log_append("assistant", "tool_use block(s) appended")

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                executor = EXECUTORS.get(block.name)
                result = executor(**block.input) if executor else f"Unknown tool: {block.name}"
                sdk_parser.log_tool_execution(block.name, block.input, result)

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})
            sdk_parser.log_append("user", f"{len(tool_results)} tool_result(s) appended")
            continue

        if response.stop_reason == "end_turn":
            final = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            print("\n[LOOP]  stop_reason='end_turn'  →  loop complete")
            if final:
                print(f"\n[MODEL REPLY]\n{final}")
            break

        print(f"\n[LOOP]  Unexpected stop_reason={response.stop_reason!r} — stopping.")
        break


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
