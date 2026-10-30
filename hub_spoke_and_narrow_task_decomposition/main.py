"""
Unified System: Hub & Spoke  +  Narrow Task Decomposition
==========================================================
Combines both patterns into one coherent workflow with three phases:

  ┌─────────────────────────────────────────────────────────────┐
  │  PHASE 1 — DECOMPOSE                                        │
  │  Hub uses "tool-as-output" to force Claude to produce a     │
  │  validated JSON task plan (narrow, specific sub-tasks).      │
  └────────────────────────┬────────────────────────────────────┘
                           │  TaskPlan (JSON)
  ┌────────────────────────▼────────────────────────────────────┐
  │  PHASE 2 — ORCHESTRATE                                      │
  │  Hub loops through tasks in dependency order.               │
  │  For each task it dispatches to exactly one Spoke Worker.   │
  │  Results are stored in ExecutionState (the shared ledger).  │
  └────────────────────────┬────────────────────────────────────┘
                           │  ExecutionState (task_id → result)
  ┌────────────────────────▼────────────────────────────────────┐
  │  PHASE 3 — AGGREGATE                                        │
  │  Hub passes all Spoke results to Claude to synthesise the   │
  │  final human-readable answer.                               │
  └─────────────────────────────────────────────────────────────┘

HOW THIS ARCHITECTURE PREVENTS ERRORS AND HALLUCINATION
────────────────────────────────────────────────────────
1. Narrow objectives stop scope creep
   Each task's `narrow_objective` is one unambiguous action.
   A Spoke that only sees "Calculate (Q3 - Q2) / Q2 × 100" cannot
   invent surrounding context or mix up which numbers to use.

2. Schema enforcement (tool-as-output) stops vague decomposition
   Claude is forced to call `submit_task_breakdown` with a strict
   JSON schema. It cannot return a loose paragraph that a parser
   might misread. Every field is typed and validated before
   Phase 2 begins.

3. Dependency injection stops hallucination of upstream data
   Before a Spoke runs, the Hub injects the verified output of
   every task it depends on into its prompt context.
   A WriterAgent that needs "the net revenue figure" receives the
   exact string the ExtractorAgent produced — it never guesses.

4. Isolated Spokes stop cross-contamination
   Spokes NEVER call each other and NEVER share a conversation
   history. Each Spoke starts with a fresh Claude context scoped
   only to its own domain. Errors stay contained.

5. ExecutionState is the single source of truth
   All results flow back through the Hub and are stored in one
   place. No Spoke is allowed to write directly to another's
   output or read results that haven't been produced yet.
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from lib import sdk_parser

MODEL     = "claude-haiku-5-5"
MAX_TURNS = 10   # safety ceiling for the aggregation loop

# Available worker roles — must match Spoke class keys registered in the Hub.
WORKER_ROLES: list[str] = [
    "extractor_worker",
    "calculator_worker",
    "writer_worker",
    "research_worker",
    "review_worker",
    "formatter_worker",
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubTask:
    """One narrow, independent unit of work produced by decomposition."""
    task_id:          str
    assigned_worker:  str
    narrow_objective: str
    expected_output:  str
    depends_on:       list[str] = field(default_factory=list)

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
    """The full decomposition output: goal restatement + ordered sub-tasks."""
    goal_summary: str
    tasks: list[SubTask]


@dataclass
class ExecutionState:
    """
    The Hub's shared ledger: maps each completed task_id to its result string.

    WHY IT EXISTS:
      Spokes never hold a reference to each other. The Hub injects each
      Spoke's output into the next Spoke's prompt via this ledger.
      This is "dependency injection without coupling."
    """
    results: dict[str, str] = field(default_factory=dict)

    def record(self, task_id: str, result: str) -> None:
        self.results[task_id] = result

    def get_context_for(self, depends_on: list[str]) -> dict[str, str]:
        """Return only the upstream results this task needs."""
        return {tid: self.results[tid] for tid in depends_on if tid in self.results}

    def is_complete(self, task_id: str) -> bool:
        return task_id in self.results

    def all_done(self, task_ids: list[str]) -> bool:
        return all(self.is_complete(tid) for tid in task_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# SPOKE AGENT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SpokeAgent(ABC):
    """
    Abstract base for all Spoke workers.

    Contract:
      handle(narrow_objective, expected_output, context) → result_str

      - narrow_objective : the single specific action to perform
      - expected_output  : the exact artifact format required
      - context          : dict of upstream task_id → result strings
                           (dependency injection — prevents hallucination)

    Spokes MUST NOT:
      - reference other Spokes
      - call the Hub
      - produce output in a format different from expected_output
    """

    @abstractmethod
    def handle(
        self,
        narrow_objective: str,
        expected_output: str,
        context: dict[str, str],
    ) -> str: ...


# ═══════════════════════════════════════════════════════════════════════════════
# CONCRETE SPOKE WORKERS
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractorWorker(SpokeAgent):
    """
    extractor_worker — extracts specific data points from a document.

    Uses mock data for the demo (a real implementation would parse a PDF/DB).
    Deterministic: same input → same output. No LLM needed here.

    WHY NO LLM: Extraction from structured data is deterministic.
    Using an LLM here would introduce hallucination risk with no benefit.
    """

    # Mock quarterly sales database (replaces a real data source)
    _MOCK_DATA: ClassVar[dict] = {
        "q2_revenue":         4_200_000,
        "q3_revenue":         5_460_000,
        "q2_units_sold":      8_400,
        "q3_units_sold":      10_500,
        "top_product":        "Wireless Headphones Pro X",
        "top_product_units":  2_100,
        "top_product_revenue": 1_050_000,
        "second_product":     "Smart Watch Series 4",
        "second_product_units": 1_800,
        "customer_regions":   {"APAC": "42%", "NA": "35%", "EU": "23%"},
        "return_rate":        "2.3%",
        "avg_order_value":    520,
    }

    def handle(
        self, narrow_objective: str, expected_output: str, context: dict[str, str]
    ) -> str:
        # Return the whole mock dataset as JSON; the Hub will pass the right
        # subset to each downstream Spoke via context injection.
        # A narrow_objective like "Extract Q3 revenue" gets the full dump so
        # the calculator and writer can pick what they need.
        return json.dumps(self._MOCK_DATA)


class CalculatorWorker(SpokeAgent):
    """
    calculator_worker — evaluates arithmetic expressions.

    Two-tier strategy:
      1. Try direct Python eval (fast, deterministic) for pure expressions.
      2. Fall back to a Claude call with context injection when the objective
         is described in natural language.

    WHY THIS PREVENTS ERRORS:
      Numbers come from ExtractorWorker via context injection, not from
      Claude's memory. Claude cannot accidentally use Q2 data for Q3 or
      invent a number it "thinks" is correct.
    """

    _SAFE_MATH: ClassVar[dict] = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    }

    _SYSTEM: ClassVar[str] = """\
You are a precise calculator. Given a numeric task and source data, perform
ONLY the computation described. Output ONLY the numeric result (and its unit
if requested). Do not explain. Use ONLY the numbers from the provided data.
"""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def handle(
        self, narrow_objective: str, expected_output: str, context: dict[str, str]
    ) -> str:
        # Tier 1: try direct eval for pure Python expressions
        for prefix in ("Calculate ", "Compute ", "Evaluate ", "Find "):
            if narrow_objective.startswith(prefix):
                expr = narrow_objective[len(prefix):].rstrip(".").strip()
                try:
                    result = eval(expr, {"__builtins__": {}}, self._SAFE_MATH)
                    return str(result)
                except (ValueError, SyntaxError, TypeError, NameError, ArithmeticError):
                    break

        # Tier 2: use Claude with injected context data
        # ANTI-HALLUCINATION: we give Claude the exact numbers from upstream tasks.
        context_block = json.dumps(context, indent=2) if context else "(no data)"
        user_msg = (
            f"Task: {narrow_objective}\n\n"
            f"Expected output: {expected_output}\n\n"
            f"Source data (use ONLY these values):\n{context_block}"
        )
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=256,
            thinking={"type": "disabled"},
            system=self._SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return next(
            (b.text for b in response.content if b.type == "text"),
            "[CalculatorWorker] No result produced.",
        )


class WriterWorker(SpokeAgent):
    """
    writer_worker — writes text artifacts from provided data.

    Uses Claude with a DOMAIN-CONSTRAINED system prompt.
    Receives upstream results via context injection, so it never
    needs to invent numbers — they are always verified.

    WHY THIS PREVENTS HALLUCINATION:
      The system prompt says "Only use the data in the context below."
      The context block contains the exact figures from the Extractor
      and Calculator. Claude cannot add numbers from its training data.
    """

    _SYSTEM: ClassVar[str] = """\
You are a specialist business writer. Your ONLY job is to write the
specific text artifact described in the task below.

Rules:
  1. Use ONLY the data provided in the "Context data" section.
  2. Do NOT invent, estimate, or round any figures.
  3. Match the format and length specified in "Expected output" exactly.
  4. Do not add introductions, sign-offs, or meta-commentary.
"""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def handle(
        self, narrow_objective: str, expected_output: str, context: dict[str, str]
    ) -> str:
        context_block = (
            json.dumps(context, indent=2) if context else "(no upstream data)"
        )
        user_msg = (
            f"Task: {narrow_objective}\n\n"
            f"Expected output format: {expected_output}\n\n"
            f"Context data (use ONLY these numbers and facts):\n{context_block}"
        )
        # Disable thinking: writer tasks don't need deep reasoning; thinking
        # blocks would consume tokens, leaving nothing for the actual text output.
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=1024,
            thinking={"type": "disabled"},
            system=self._SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return next(
            (b.text for b in response.content if b.type == "text"),
            "[WriterWorker] No output produced.",
        )


class ResearchWorker(SpokeAgent):
    """
    research_worker — looks up benchmark or industry data.

    Uses mock benchmarks (a real implementation would call a search API).
    Deterministic mock prevents LLM hallucination of "industry averages."
    """

    _BENCHMARKS: ClassVar[dict] = {
        "ecommerce_qoq_growth_avg":    "18%",
        "ecommerce_return_rate_avg":   "3.5%",
        "electronics_avg_order_value": "$480",
        "top_growth_category_2024":    "Wearables (+34% YoY)",
        "apac_market_share_trend":     "APAC gaining 4pp annually",
    }

    def handle(
        self, narrow_objective: str, expected_output: str, context: dict[str, str]
    ) -> str:
        return json.dumps(self._BENCHMARKS)


class ReviewWorker(SpokeAgent):
    """
    review_worker — validates the output of a writing or formatting task.

    Uses Claude to check for inconsistencies between the written text
    and the source data. Returns a structured review report.

    WHY IT EXISTS:
      An isolated review step with access to BOTH the written output
      AND the raw source data catches cases where the WriterWorker
      drifted from the numbers. This is a final hallucination gate.
    """

    _SYSTEM: ClassVar[str] = """\
You are a fact-checker for business reports. You receive:
  - A written report
  - The source data the report was based on

Your job: identify any factual inconsistencies between the report
and the source data. Be concise. Format your response as:

VERDICT: PASS | FAIL
ISSUES: <bullet list of issues, or "None" if PASS>
"""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def handle(
        self, narrow_objective: str, expected_output: str, context: dict[str, str]
    ) -> str:
        context_block = json.dumps(context, indent=2) if context else "(no source data)"
        user_msg = (
            f"Task: {narrow_objective}\n\n"
            f"Source data:\n{context_block}"
        )
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=512,
            thinking={"type": "disabled"},
            system=self._SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return next(
            (b.text for b in response.content if b.type == "text"),
            "[ReviewWorker] No verdict produced.",
        )


class FormatterWorker(SpokeAgent):
    """
    formatter_worker — formats the final combined output.

    Uses Claude to structure and polish the aggregated content.
    Receives all upstream outputs via context injection.
    """

    _SYSTEM: ClassVar[str] = """\
You are a document formatter. You receive sections of a business report
and must combine them into a clean, professional final document.

Rules:
  1. Use only the provided sections — do not add new content.
  2. Preserve all numbers and facts exactly as given.
  3. Use clear headings and bullet points where appropriate.
"""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def handle(
        self, narrow_objective: str, expected_output: str, context: dict[str, str]
    ) -> str:
        context_block = json.dumps(context, indent=2) if context else "(no sections)"
        user_msg = (
            f"Task: {narrow_objective}\n\n"
            f"Expected output format: {expected_output}\n\n"
            f"Report sections to combine:\n{context_block}"
        )
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=2048,
            thinking={"type": "disabled"},
            system=self._SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return next(
            (b.text for b in response.content if b.type == "text"),
            "[FormatterWorker] No output produced.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED HUB
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedHub:
    """
    The central coordinator. Owns all Spoke instances and drives all three phases.

    ANTI-HALLUCINATION DESIGN SUMMARY:
      Phase 1: Schema-enforced JSON decomposition → no vague tasks
      Phase 2: Dependency injection → Spokes only see verified upstream data
               Isolated Spoke contexts → errors stay contained
      Phase 3: All results in ExecutionState → aggregation uses facts only
    """

    # ── Phase 1 system prompt ─────────────────────────────────────────────────
    _DECOMPOSE_SYSTEM: ClassVar[str] = f"""\
You are a task decomposition coordinator.

Break the user's complex goal into ultra-specific narrow sub-tasks.

Rules:
  1. Each narrow_objective is ONE action — no "and", no "then".
  2. expected_output is an exact artifact type with field names or length.
  3. assigned_worker must be one of: {WORKER_ROLES}
     - extractor_worker : extract raw data from a data source
     - calculator_worker: evaluate a numeric expression
     - writer_worker    : produce written text from provided data
     - research_worker  : look up benchmarks or external facts
     - review_worker    : validate output against source data
     - formatter_worker : combine and format sections into a document
  4. depends_on lists task_ids whose results this task needs injected.
  5. Maximise parallelism: independent tasks get empty depends_on.

Produce 5–8 sub-tasks. Call `submit_task_breakdown`.
"""

    # ── Phase 1 tool (forces structured JSON output) ──────────────────────────
    _DECOMPOSE_TOOL: ClassVar[dict] = {
        "name": "submit_task_breakdown",
        "description": "Submit the task breakdown. Do not respond in plain text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal_summary": {"type": "string"},
                "tasks": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_id":          {"type": "string"},
                            "assigned_worker":  {"type": "string", "enum": WORKER_ROLES},
                            "narrow_objective": {"type": "string"},
                            "expected_output":  {"type": "string"},
                            "depends_on":       {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "task_id", "assigned_worker", "narrow_objective",
                            "expected_output", "depends_on",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["goal_summary", "tasks"],
            "additionalProperties": False,
        },
    }

    # ── Phase 3 system prompt ─────────────────────────────────────────────────
    _AGGREGATE_SYSTEM: ClassVar[str] = """\
You are a report synthesiser. You receive all task results from a multi-agent
workflow and must produce a clear, concise final answer for the user.

Rules:
  1. Use ONLY the task results provided — do not add invented data.
  2. If a review task produced a FAIL verdict, mention it.
  3. Format the final answer as a professional executive response.
"""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

        # ── Spoke registry: ONLY the Hub holds these references ───────────────
        # This is where the Hub-and-Spoke rule is enforced in code.
        # No Spoke is ever handed a reference to another Spoke.
        self._spokes: dict[str, SpokeAgent] = {
            "extractor_worker":  ExtractorWorker(),
            "calculator_worker": CalculatorWorker(),
            "writer_worker":     WriterWorker(),
            "research_worker":   ResearchWorker(),
            "review_worker":     ReviewWorker(),
            "formatter_worker":  FormatterWorker(),
        }

    # ── PHASE 1: DECOMPOSE ────────────────────────────────────────────────────

    def _phase1_decompose(self, user_goal: str) -> TaskPlan:
        """
        Ask Claude to break the goal into a validated JSON task plan.
        Uses tool-as-output to guarantee schema compliance.
        """
        _banner("PHASE 1 — NARROW TASK DECOMPOSITION")
        print(f"\n[HUB]  Goal: {user_goal!r}")
        print("[HUB]  Sending to Claude with forced tool_use schema …")

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=self._DECOMPOSE_SYSTEM,
            tools=[self._DECOMPOSE_TOOL],
            tool_choice={"type": "tool", "name": "submit_task_breakdown"},
            messages=[{"role": "user", "content": user_goal}],
        )

        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            raise RuntimeError("Claude did not call submit_task_breakdown.")

        raw: dict = tool_block.input
        _section("TASK PLAN JSON")
        print(json.dumps(raw, indent=2, ensure_ascii=False))

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

    # ── PHASE 2: ORCHESTRATE ──────────────────────────────────────────────────

    def _phase2_orchestrate(
        self, plan: TaskPlan, state: ExecutionState
    ) -> None:
        """
        Execute every sub-task by dispatching to Spokes in dependency order.

        DEPENDENCY ORDER ALGORITHM:
          Loop until all tasks are done. Each iteration executes every task
          whose dependencies are all satisfied. This handles arbitrary DAGs
          (not just linear chains) and maximises parallelism opportunities.

        DEPENDENCY INJECTION:
          Before calling a Spoke, the Hub fetches the results of its
          declared dependencies from ExecutionState and passes them as
          the `context` dict. The Spoke ONLY sees these upstream results —
          it cannot read any other task's output.
        """
        _banner("PHASE 2 — HUB & SPOKE ORCHESTRATION")
        print(f"\n[HUB]  Dispatching {len(plan.tasks)} task(s) in dependency order …")

        pending = list(plan.tasks)
        wave = 0

        # Loop until all tasks are executed
        while pending:
            # Find every task whose dependencies are all satisfied
            ready = [
                t for t in pending
                if state.all_done(t.depends_on)
            ]
            if not ready:
                stuck = [t.task_id for t in pending]
                raise RuntimeError(
                    f"Dependency cycle or missing task_id. Stuck: {stuck}"
                )

            wave += 1
            _section(f"EXECUTION WAVE {wave}  ({len(ready)} task(s) ready)")

            for task in ready:
                # ── Dependency injection ──────────────────────────────────────
                # ANTI-HALLUCINATION: the Spoke sees only verified upstream data.
                context = state.get_context_for(task.depends_on)

                print(f"\n  ┌─ [{task.task_id}]  {task.assigned_worker.upper()}")
                print(f"  │  Objective : {task.narrow_objective}")
                print(f"  │  Expected  : {task.expected_output}")
                if context:
                    deps_preview = {k: v[:60] + "…" if len(v) > 60 else v
                                    for k, v in context.items()}
                    print(f"  │  Context   : {deps_preview}  ← injected from upstream")
                else:
                    print("  │  Context   : (none — independent task)")

                # ── Dispatch to the Spoke (STRICT: Hub is the only dispatcher) ─
                spoke = self._spokes.get(task.assigned_worker)
                if spoke is None:
                    result = f"[HUB ERROR] No spoke registered for {task.assigned_worker!r}"
                else:
                    result = spoke.handle(
                        narrow_objective=task.narrow_objective,
                        expected_output=task.expected_output,
                        context=context,
                    )

                # ── Store result in ExecutionState ────────────────────────────
                state.record(task.task_id, result)
                result_preview = result[:80] + "…" if len(result) > 80 else result
                print(f"  │  Result    : {result_preview}")
                print(f"  └─ [DONE]  {task.task_id} → stored in ExecutionState")

                pending.remove(task)

        _section("ORCHESTRATION COMPLETE")
        print(f"  All {len(plan.tasks)} tasks executed. ExecutionState has {len(state.results)} result(s).")

    # ── PHASE 3: AGGREGATE ────────────────────────────────────────────────────

    def _phase3_aggregate(self, plan: TaskPlan, state: ExecutionState) -> str:
        """
        Pass all task results to Claude for final synthesis.

        The aggregation prompt contains ONLY the verified results from
        ExecutionState — no invented data can enter at this stage.
        """
        _banner("PHASE 3 — AGGREGATION")
        print("\n[HUB]  Building aggregation prompt from ExecutionState …")

        # Build a structured summary of all task results
        results_block = "\n\n".join(
            f"[{task.task_id}] {task.assigned_worker}\n"
            f"Objective: {task.narrow_objective}\n"
            f"Result:\n{state.results.get(task.task_id, '(no result)')}"
            for task in plan.tasks
        )

        # Single call with enough tokens. Aggregation is one-shot:
        # no tools, no continuation needed. Return whatever text we get,
        # even if stop_reason is max_tokens (partial answer is still useful).
        sdk_parser.log_turn_header(1)

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=2048,
            thinking={"type": "disabled"},
            system=self._AGGREGATE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Original goal: {plan.goal_summary}\n\n"
                        f"All task results:\n{results_block}\n\n"
                        "Please synthesise a final executive answer for the user."
                    ),
                }
            ],
        )

        sdk_parser.log_stop_reason(response)
        sdk_parser.log_content_blocks(response)
        sdk_parser.log_divider()

        final = next((b.text for b in response.content if b.type == "text"), "")
        if not final:
            return "[HUB] Aggregation produced no text output."
        return final

    # ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────

    def run(self, user_goal: str) -> str:
        """
        Full unified workflow: Decompose → Orchestrate → Aggregate.
        Returns the final answer string.
        """
        _banner("UNIFIED HUB  —  START")
        print(f"  Model   : {MODEL}")
        print(f"  Spokes  : {list(self._spokes)}")
        print(f"\n[USER → HUB]  {user_goal!r}")

        # ── Phase 1: break the goal into narrow tasks ─────────────────────────
        plan = self._phase1_decompose(user_goal)

        # ── Phase 2: execute each task via the correct Spoke ──────────────────
        state = ExecutionState()
        self._phase2_orchestrate(plan, state)

        # ── Phase 3: synthesise the final answer from all results ─────────────
        final = self._phase3_aggregate(plan, state)

        _banner("FINAL ANSWER TO USER")
        print(final)
        print("═" * 62)
        return final


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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
    hub = UnifiedHub()
    hub.run(
        "Analyze our e-commerce store's Q3 2024 sales performance compared to Q2, "
        "identify the top-performing product, benchmark our growth against industry "
        "averages, and produce a brief executive report."
    )


if __name__ == "__main__":
    main()
