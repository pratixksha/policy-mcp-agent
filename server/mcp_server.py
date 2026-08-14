import json
import os
from datetime import datetime, timezone

try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2.0
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp < 2.0

RULES_PATH = os.path.join(os.path.dirname(__file__), "policy_rules.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "decisions_log.jsonl")

mcp = _Server("policy-checker")


def _load_rules():
    with open(RULES_PATH) as f:
        return json.load(f)["rules"]


@mcp.tool()
def list_policy_rules() -> str:
    """Return the full list of content policy rules (id, category, description)."""
    return json.dumps(_load_rules(), indent=2)


@mcp.tool()
def check_against_policy(text: str, rule_id: str) -> str:
    rules = {r["rule_id"]: r for r in _load_rules()}
    if rule_id not in rules:
        return json.dumps({"error": f"Unknown rule_id: {rule_id}"})

    rule = rules[rule_id]
    lowered = text.lower()

    keyword_banks = {
        "R1_HARASSMENT": ["idiot", "kill yourself", "shut up", "worthless", "hate you", "ugly", "stupid"],
        "R2_SPAM": ["click here", "free money", "act now", "subscribe now", "follow for follow", "dm me", "limited time offer"],
        "R3_MISINFORMATION": ["cures cancer", "vaccines cause", "proven fact", "doctors don't want you to know", "secretly"],
        "R4_SELF_HARM": ["want to die", "end my life", "how to hurt myself", "no reason to live"],
        "R5_SAFE": [],
    }

    matched = [kw for kw in keyword_banks.get(rule_id, []) if kw in lowered]
    signal_strength = min(1.0, 0.3 + 0.35 * len(matched)) if matched else 0.05

    result = {
        "rule_id": rule_id,
        "category": rule["category"],
        "matched_keywords": matched,
        "heuristic_signal_strength": round(signal_strength, 2),
        "note": "Heuristic signal only — the agent should weigh this alongside its own reading of the text.",
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def log_decision(text: str, decision: str, rule_id: str, confidence: float, rationale: str) -> str:
    """
    Log a final moderation decision to a local JSONL file for later evaluation.

    Args:
        text: the original content being moderated
        decision: "VIOLATION" or "SAFE"
        rule_id: the rule_id that applies (e.g. "R1_HARASSMENT", or "R5_SAFE" if safe)
        confidence: agent's confidence in this decision, 0.0-1.0
        rationale: one or two sentence explanation
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "decision": decision,
        "rule_id": rule_id,
        "confidence": confidence,
        "rationale": rationale,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return json.dumps({"status": "logged", "entry": entry})


if __name__ == "__main__":
    mcp.run()
