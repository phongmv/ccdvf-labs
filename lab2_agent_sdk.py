import asyncio
import json
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
)


async def on_pre_tool_use(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    # Fires before a tool runs: lets you see/gate the call
    print(f"[PRE_HOOK] tool={input_data['tool_name']} id={tool_use_id}")
    print(f"       input={json.dumps(input_data['tool_input'])}")
    print("[PRE_HOOK] is working!")
    return {}  # {} = allow unchanged


async def on_post_tool_use(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    # Fires after a tool returns: lets you see its result
    print(f"[POST] tool={input_data['tool_name']} id={tool_use_id}")
    print(
        f"       response={json.dumps(input_data['tool_response'], default=str)[:300]}"
    )
    print("[POST_HOOK] is working!")
    return {}


async def main():
    # Agentic loop: streams messages as Claude works
    async for message in query(
        prompt="Run `echo hello-from-hook` with Bash and tell me the output.",
        options=ClaudeAgentOptions(
            allowed_tools=["Bash"],  # Bash triggers tool hooks
            permission_mode="acceptEdits",
            hooks={
                "PreToolUse": [HookMatcher(hooks=[on_pre_tool_use])],
                "PostToolUse": [HookMatcher(hooks=[on_post_tool_use])],
            },
        ),
    ):
        if isinstance(message, ResultMessage):
            print(f"\n[RESULT] {message.result}")
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                print(f"[ASSISTANT] {block}")


asyncio.run(main())
