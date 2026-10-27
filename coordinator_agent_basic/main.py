"""
Coordinator Agent — Basic Pattern
===================================
One coordinator agent receives a task and delegates subtasks to specialized
worker agents. Each worker is exposed as a tool — the coordinator decides
which workers to call, in what order, and passes results between them.

Architecture:
  User
   └─► Coordinator (this loop)
          ├─► research_worker   — gathers raw information
          ├─► analysis_worker   — extracts key insights from raw data
          └─► writing_worker    — formats the final deliverable

The coordinator never does the work itself; it only orchestrates.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from lib import sdk_parser

MODEL = "claude-haiku-5-5"
MAX_ITERATIONS = 10

# ── System prompt — coordinator role ─────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a coordinator agent. Your job is to break a task into subtasks and
delegate each one to the right worker tool. You do not do the work yourself.

Workflow:
  1. Call research_worker first to gather raw information on the topic.
  2. Call analysis_worker with the research output to extract key insights.
  3. Call writing_worker with the analysis output to produce the final deliverable.
  4. When all workers are done, summarise the final result for the user.

Always follow this sequence. Do not skip workers or combine steps.
"""

# ── Worker tool definitions ───────────────────────────────────────────────────

WORKER_TOOLS = [
    {
        "name": "research_worker",
        "description": (
            "Gather raw information about a topic. "
            "Call this first, before any analysis or writing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The subject to research.",
                },
                "focus": {
                    "type": "string",
                    "description": "Specific angle or question to focus on.",
                },
            },
            "required": ["topic", "focus"],
        },
    },
    {
        "name": "analysis_worker",
        "description": (
            "Extract key insights from raw research data. "
            "Call this after research_worker, passing its output as input."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "raw_data": {
                    "type": "string",
                    "description": "The raw research output to analyse.",
                },
                "analysis_goal": {
                    "type": "string",
                    "description": "What the analysis should highlight or answer.",
                },
            },
            "required": ["raw_data", "analysis_goal"],
        },
    },
    {
        "name": "writing_worker",
        "description": (
            "Format insights into a polished deliverable. "
            "Call this last, after analysis_worker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "insights": {
                    "type": "string",
                    "description": "The analysis output to turn into a deliverable.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["executive_summary", "bullet_list", "report"],
                    "description": "Format of the final deliverable.",
                },
            },
            "required": ["insights", "output_format"],
        },
    },
]

# ── Mock worker implementations ───────────────────────────────────────────────
# In production these would be real API calls, sub-agent invocations,
# database lookups, or calls to other Claude sessions.

def research_worker(topic: str, focus: str) -> str:
    return (
        f"[RESEARCH OUTPUT — topic: {topic!r}, focus: {focus!r}]\n"
        "• Global EV sales reached 14 million units in 2023, up 35% YoY.\n"
        "• Battery costs fell 90% over the last decade; now ~$130/kWh.\n"
        "• Top markets: China (60%), Europe (25%), USA (10%).\n"
        "• Key players: BYD, Tesla, Volkswagen, SAIC, Stellantis.\n"
        "• Main barriers: charging infrastructure, range anxiety, upfront cost.\n"
        "• Governments offering purchase subsidies and mandating ICE phase-outs by 2035."
    )


def analysis_worker(raw_data: str, analysis_goal: str) -> str:
    return (
        f"[ANALYSIS OUTPUT — goal: {analysis_goal!r}]\n"
        "Key insight 1: Market is in hyper-growth but concentrated — China alone "
        "drives 60% of volume, creating geographic risk for Western manufacturers.\n"
        "Key insight 2: Battery cost trajectory suggests price parity with ICE "
        "vehicles by 2026-2027, which will be the inflection point for mass adoption.\n"
        "Key insight 3: Infrastructure gap is the critical bottleneck; sales growth "
        "is outpacing charger deployment 3:1 in most markets.\n"
        "Key insight 4: BYD's vertical integration (cells → vehicles) gives it a "
        "structural cost advantage of ~$3,000/vehicle over Western OEMs."
    )


def writing_worker(insights: str, output_format: str) -> str:
    if output_format == "bullet_list":
        return (
            "[FINAL DELIVERABLE — bullet_list]\n"
            "• EV market growing 35% YoY; 14M units sold in 2023.\n"
            "• Price parity with ICE vehicles expected by 2026-2027.\n"
            "• China dominates at 60% share — geographic concentration risk.\n"
            "• Charging infrastructure is the #1 adoption bottleneck.\n"
            "• BYD leads on cost; $3k/vehicle structural advantage over Western OEMs."
        )
    if output_format == "executive_summary":
        return (
            "[FINAL DELIVERABLE — executive_summary]\n"
            "The EV market is growing rapidly (35% YoY, 14M units in 2023) but "
            "remains heavily concentrated in China. Price parity with combustion "
            "vehicles is within two years, driven by falling battery costs. The "
            "critical near-term constraint is charging infrastructure, not vehicle "
            "technology. BYD's vertical integration positions it as the low-cost "
            "leader; Western OEMs must respond through partnerships or acquisitions."
        )
    return (
        "[FINAL DELIVERABLE — report]\n"
        "## EV Market Overview\n"
        "Sales reached 14M units in 2023 (+35% YoY), led by China (60% share).\n\n"
        "## Cost Trajectory\n"
        "Battery costs are at ~$130/kWh, down 90% over a decade. "
        "ICE price parity expected 2026-2027.\n\n"
        "## Key Risks\n"
        "- Geographic concentration in China\n"
        "- Charging infrastructure lagging sales growth 3:1\n\n"
        "## Competitive Landscape\n"
        "BYD holds a ~$3,000/vehicle cost advantage through vertical integration."
    )


WORKERS = {
    "research_worker": research_worker,
    "analysis_worker": analysis_worker,
    "writing_worker":  writing_worker,
}

# ── Coordinator loop ──────────────────────────────────────────────────────────

def run(user_task: str) -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_task}]

    print(f"\n{'═' * 64}")
    print("  Coordinator Agent — Basic")
    print(f"  Model         : {MODEL}")
    print(f"  MAX_ITERATIONS: {MAX_ITERATIONS}")
    print(f"{'═' * 64}")
    print(f"\n[USER TASK]  {user_task}")

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        sdk_parser.log_turn_header(iteration)
        print(f"  [iteration {iteration}/{MAX_ITERATIONS}]")
        sdk_parser.log_messages_state(messages)

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=WORKER_TOOLS,
            messages=messages,
        )

        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)
        sdk_parser.log_divider()

        # ── Coordinator finished ─────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            final = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            print(f"\n[COORDINATOR]  All workers done — iteration {iteration}/{MAX_ITERATIONS}")
            print(f"\n{'─' * 64}")
            print("[FINAL DELIVERABLE TO USER]")
            print(final)
            print(f"{'─' * 64}")
            break

        # ── Coordinator delegating to a worker ──────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            sdk_parser.log_append("assistant", "coordinator delegated to worker(s)")

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                worker = WORKERS.get(block.name)
                result = worker(**block.input) if worker else f"Unknown worker: {block.name}"

                print(f"\n  ┌─ [{block.name.upper()}]")
                for line in result.splitlines():
                    print(f"  │  {line}")
                print("  └─ done")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})
            sdk_parser.log_append(
                "user", f"{len(tool_results)} worker result(s) returned to coordinator"
            )
            continue

        print(f"\n[COORDINATOR]  Unexpected stop_reason={response.stop_reason!r} — stopping.")
        break

    else:
        print(f"\n[COORDINATOR]  MAX_ITERATIONS ({MAX_ITERATIONS}) reached — stopping.")

    print(f"\n[EXIT]  iterations_used={iteration}/{MAX_ITERATIONS}\n")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run("Prepare an executive summary on the global electric vehicle market.")
