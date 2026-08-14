import argparse
import asyncio
import json
import os
import sys

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "server", "mcp_server.py")

SYSTEM_PROMPT = """You are a content moderation assistant. You are given a piece of
user-generated text and must decide whether it violates platform policy.

You have access to tools:
- list_policy_rules: see the full rule set
- check_against_policy: run a heuristic check of the text against a specific rule_id
- log_decision: record your final decision

Process:
1. Call list_policy_rules to see the categories.
2. Consider which rule(s) might apply, and call check_against_policy for the
   most plausible one(s). The heuristic signal is a weak input, not the final
   answer — use your own judgment about the actual meaning and context of the text.
3. Call log_decision exactly once with your final decision: "VIOLATION" or "SAFE",
   the rule_id that applies (use "R5_SAFE" if the text is safe), a confidence
   score (0.0-1.0), and a one-sentence rationale.

Be precise — do not flag content as a violation unless it clearly matches a
rule. Ambiguous or merely negative/critical content that doesn't target,
threaten, or mislead anyone should be marked SAFE.
"""


async def classify(text: str, model: str = "claude-sonnet-4-5") -> dict:
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            claude_tools = []
            for t in tools_result.tools:
                schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
                claude_tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": schema,
                })

            messages = [{"role": "user", "content": f"Moderate this text:\n\n{text}"}]
            final_decision = None

            for _ in range(6):
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=claude_tools,
                    messages=messages,
                )

                messages.append({"role": "assistant", "content": response.content})

                tool_calls = [b for b in response.content if b.type == "tool_use"]
                if not tool_calls:
                    break

                tool_results = []
                for call in tool_calls:
                    result = await session.call_tool(call.name, call.input)
                    result_text = "".join(
                        block.text for block in result.content if hasattr(block, "text")
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result_text,
                    })
                    if call.name == "log_decision":
                        final_decision = json.loads(result_text)["entry"]

                messages.append({"role": "user", "content": tool_results})

                if final_decision:
                    break

            return final_decision or {"decision": "ERROR", "rule_id": None, "rationale": "Agent did not log a decision."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to classify")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment first.")
        sys.exit(1)

    result = asyncio.run(classify(args.text))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
