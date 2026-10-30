"""
Narrow Task Decomposition — AI Agent Coordinator Pattern
=========================================================
Core concept:
  A Coordinator receives a complex, ambiguous user goal and uses Claude to
  break it down into ultra-specific, narrow, independent sub-tasks.

  Each sub-task is machine-readable JSON with a strict schema so downstream
  workers can execute them without ambiguity.

Why "narrow"?
  Broad goal  → "Analyze the Q3 financial report and write a summary"
  Narrow task → "Extract the net revenue figure from page 4 of the PDF"
                "Calculate the YoY revenue growth percentage using the formula
                 (current - previous) / previous * 100"
                "Write a 3-sentence executive summary citing exact figures"

Architecture:
                          ┌──────────────────────────────────────────┐
  Complex Goal ──────────►│       TaskDecompositionCoordinator        │
                          │  Uses Claude + strict tool schema to       │
                          │  output a validated JSON task breakdown    │
                          └────────────────┬─────────────────────────┘
                                           │
                          ┌────────────────▼──────────────────────────┐
                          │           Task Plan (JSON)                 │
                          │  [                                         │
                          │    { task_id, assigned_worker,             │
                          │      narrow_objective, expected_output,    │
                          │      depends_on },                         │
                          │    ...                                     │
                          │  ]                                         │
                          └───────────────────────────────────────────┘

Key technique: tool_use with strict schema
  We define a tool `submit_task_breakdown` with a rigid JSON schema.
  Claude MUST call this tool to respond, which forces structured output.
  This is called "tool-as-output" — using tool_use purely to enforce JSON.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from lib import sdk_parser

MODEL = "claude-haiku-5-5"

# ── Available workers (the "spoke" roles in this system) ──────────────────────
AVAILABLE_WORKERS = [
    "data_extraction_worker",
    "financial_analysis_worker",
    "writing_worker",
    "review_worker",
    "research_worker",
    "calculation_worker",
    "formatting_worker",
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA SCHEMA
# Each sub-task must conform to this shape.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubTask:
    task_id:         str
    assigned_worker: str
    narrow_objective: str
    expected_output: str
    depends_on:      list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id":          self.task_id,
            "assigned_worker":  self.assigned_worker,
            "narrow_objective": self.narrow_objective,
            "expected_output":  self.expected_output,
            "depends_on":       self.depends_on,
        }


@dataclass
class TaskPlan:
    goal_summary: str
    tasks: list[SubTask]

    def to_dict(self) -> dict:
        return {
            "goal_summary": self.goal_summary,
            "tasks": [t.to_dict() for t in self.tasks],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TaskDecompositionCoordinator:
    """
    Coordinator that decomposes a complex user goal into narrow, specific
    sub-tasks using Claude with a strict JSON output schema.

    Technique: "tool-as-output"
      We define a tool whose only purpose is to RECEIVE Claude's answer in
      structured form.  By setting tool_choice={"type": "tool"}, Claude is
      FORCED to call this tool — it cannot respond with free text.
      This gives us guaranteed, validated JSON every time.
    """

    # ── System prompt ─────────────────────────────────────────────────────────
    _SYSTEM: ClassVar[str] = f"""\
You are a task decomposition coordinator for an AI agent system.

Your job: take a complex user goal and break it into multiple ultra-specific,
narrow, independent sub-tasks that can be executed by specialist workers.

Rules for each sub-task:
  1. narrow_objective must be ONE specific action — not "analyse and summarise",
     but "extract the net revenue figure from section 2.1".
  2. expected_output must be an exact artifact: a number, a string of N words,
     a JSON object with specific keys, etc.
  3. assigned_worker must come from this list: {AVAILABLE_WORKERS}
  4. depends_on lists task_ids that must complete before this task starts.
     Tasks with no dependencies can run in parallel.
  5. Prefer parallelism: if two tasks are independent, give them empty depends_on.

Produce at least 4 and at most 8 sub-tasks per goal.
Call `submit_task_breakdown` with your result.
"""

    # ── Tool schema that enforces structured output ───────────────────────────
    # Claude MUST call this tool — it cannot respond any other way.
    _DECOMPOSE_TOOL: ClassVar[dict] = {
        "name": "submit_task_breakdown",
        "description": (
            "Submit the decomposed task breakdown as structured JSON. "
            "Call this tool with the full breakdown — do not respond in plain text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal_summary": {
                    "type": "string",
                    "description": "One-sentence restatement of the user goal.",
                },
                "tasks": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 8,
                    "description": "List of narrow, specific sub-tasks.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Unique ID, e.g. 'T01', 'T02'.",
                            },
                            "assigned_worker": {
                                "type": "string",
                                "enum": AVAILABLE_WORKERS,
                                "description": "Which specialist worker handles this.",
                            },
                            "narrow_objective": {
                                "type": "string",
                                "description": (
                                    "One specific, unambiguous action. "
                                    "No conjunctions ('and', 'then'). "
                                    "Must be completable in isolation."
                                ),
                            },
                            "expected_output": {
                                "type": "string",
                                "description": (
                                    "The exact artifact this task produces: "
                                    "a number, a string, a JSON object, etc."
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "task_ids that must finish before this starts. "
                                    "Empty list = can run immediately."
                                ),
                            },
                        },
                        "required": [
                            "task_id",
                            "assigned_worker",
                            "narrow_objective",
                            "expected_output",
                            "depends_on",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["goal_summary", "tasks"],
            "additionalProperties": False,
        },
    }

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    # ── Main public method ────────────────────────────────────────────────────

    def decompose(self, user_goal: str) -> TaskPlan:
        """
        Takes a complex user goal and returns a validated TaskPlan.

        Steps:
          1. Send goal to Claude with the strict tool schema.
          2. Claude is forced (tool_choice) to call submit_task_breakdown.
          3. Parse and validate the JSON from the tool_use block.
          4. Return a TaskPlan dataclass.
        """
        _banner("NARROW TASK DECOMPOSITION COORDINATOR")
        print(f"  Model   : {MODEL}")
        print(f"  Workers : {AVAILABLE_WORKERS}")
        print(f"\n[GOAL IN]  {user_goal!r}")

        _section("TECHNIQUE: TOOL-AS-OUTPUT")
        print("  We define 'submit_task_breakdown' as a tool.")
        print("  tool_choice = {type: tool, name: submit_task_breakdown}")
        print("  → Claude CANNOT respond in free text.")
        print("  → It MUST call the tool with valid JSON matching our schema.")
        print("  → This guarantees structured output every time.")

        messages = [{"role": "user", "content": user_goal}]

        sdk_parser.log_turn_header(1)
        sdk_parser.log_messages_state(messages)

        # ── Send to Claude; force it to call our structured tool ──────────────
        print("\n[COORDINATOR → CLAUDE]  requesting task breakdown (forced tool call) …")
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=self._SYSTEM,
            tools=[self._DECOMPOSE_TOOL],
            tool_choice={            # ← KEY: forces Claude to call this exact tool
                "type": "tool",
                "name": "submit_task_breakdown",
            },
            messages=messages,
        )

        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)
        sdk_parser.log_divider()

        # ── Extract the tool_use block ────────────────────────────────────────
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            raise RuntimeError("Claude did not call submit_task_breakdown — unexpected.")

        raw: dict = tool_block.input   # already parsed JSON (dict), not a string

        _section("RAW JSON FROM CLAUDE (tool_use.input)")
        print(json.dumps(raw, indent=2, ensure_ascii=False))

        # ── Build typed TaskPlan from the raw dict ────────────────────────────
        plan = self._build_plan(raw)
        return plan

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_plan(raw: dict) -> TaskPlan:
        tasks = [
            SubTask(
                task_id=t["task_id"],
                assigned_worker=t["assigned_worker"],
                narrow_objective=t["narrow_objective"],
                expected_output=t["expected_output"],
                depends_on=t.get("depends_on", []),
            )
            for t in raw["tasks"]
        ]
        return TaskPlan(goal_summary=raw["goal_summary"], tasks=tasks)


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def display_plan(plan: TaskPlan) -> None:
    """Pretty-print the task plan with dependency tree."""

    _section("TASK PLAN SUMMARY")
    print(f"  Goal : {plan.goal_summary}")
    print(f"  Tasks: {len(plan.tasks)}")

    # ── Table view ────────────────────────────────────────────────────────────
    _section("TASK BREAKDOWN TABLE")
    col_w = [6, 28, 38, 32, 16]
    header = (
        f"{'ID':<{col_w[0]}} "
        f"{'Worker':<{col_w[1]}} "
        f"{'Narrow Objective':<{col_w[2]}} "
        f"{'Expected Output':<{col_w[3]}} "
        f"{'Depends On':<{col_w[4]}}"
    )
    print(header)
    print("─" * sum(col_w))
    for t in plan.tasks:
        obj = t.narrow_objective[:37] + "…" if len(t.narrow_objective) > 37 else t.narrow_objective
        out = t.expected_output[:31] + "…" if len(t.expected_output) > 31 else t.expected_output
        deps = ", ".join(t.depends_on) if t.depends_on else "—"
        print(
            f"{t.task_id:<{col_w[0]}} "
            f"{t.assigned_worker:<{col_w[1]}} "
            f"{obj:<{col_w[2]}} "
            f"{out:<{col_w[3]}} "
            f"{deps:<{col_w[4]}}"
        )

    # ── Dependency / parallelism view ─────────────────────────────────────────
    _section("DEPENDENCY ANALYSIS (what can run in parallel)")
    id_set = {t.task_id for t in plan.tasks}
    parallel = [t for t in plan.tasks if not t.depends_on]
    sequential = [t for t in plan.tasks if t.depends_on]

    print("\n  ▶ PARALLEL (no dependencies — start immediately):")
    for t in parallel:
        print(f"      {t.task_id}  [{t.assigned_worker}]  {t.narrow_objective[:60]}")

    if sequential:
        print("\n  ▶ SEQUENTIAL (must wait for dependencies):")
        for t in sequential:
            blocking = [d for d in t.depends_on if d in id_set]
            print(f"      {t.task_id}  [{t.assigned_worker}]  waits for {blocking}")
            print(f"           → {t.narrow_objective[:60]}")

    # ── Full JSON output ──────────────────────────────────────────────────────
    _section("FINAL TASK PLAN JSON")
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))


def _banner(title: str) -> None:
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")


def _section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 58 - len(title))}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    coordinator = TaskDecompositionCoordinator()

    goals = [
        # Goal 1: financial analysis report
        "Analyze last quarter's financial report and write a brief executive summary.",

        # Goal 2: product launch planning
        "Plan and execute a social media campaign for launching our new mobile app.",
    ]

    for i, goal in enumerate(goals, 1):
        _banner(f"DEMO {i} / {len(goals)}")
        plan = coordinator.decompose(goal)
        display_plan(plan)
        print()


if __name__ == "__main__":
    main()
