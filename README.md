# Policy Check Agent — MCP + Claude Tool Use

A content-moderation agent built on the **Model Context Protocol (MCP)**: a small MCP server exposes policy-checking tools, and a Claude-powered agent calls those tools to classify text and log its decisions — then gets evaluated against a labeled test set with precision, recall, and false-positive-rate.

## The idea

Real moderation pipelines combine several weak signals (keyword heuristics, classifier scores, human review) rather than relying on one black-box model. This project mirrors that pattern at small scale: the MCP server exposes a transparent, rule-by-rule heuristic checker as a **tool**, and Claude acts as the reasoning layer — deciding which rules to check, weighing the heuristic signal against its own reading of the text, and logging a final, explainable decision.

## Architecture

```
server/mcp_server.py     → MCP server exposing 3 tools:
                              - list_policy_rules
                              - check_against_policy(text, rule_id)
                              - log_decision(text, decision, rule_id, confidence, rationale)
server/policy_rules.json → the rule set (harassment, spam, misinformation, self-harm, safe)

agent/agent.py            → connects to the MCP server over stdio, gives Claude
                             the tools, and runs an agent loop until Claude logs
                             a final decision

eval/eval.py               → runs the agent against data/test_set.json (20 labeled
                              examples) and computes precision / recall / FPR / accuracy

data/test_set.json         → hand-written labeled examples across all 5 categories
data/decisions_log.jsonl   → append-only log of every decision the agent has logged
data/eval_results.json     → output of the last eval run
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

**Classify a single piece of text:**
```bash
python agent/agent.py --text "you are so stupid, shut up"
```

**Run the full evaluation:**
```bash
python eval/eval.py
```

This prints precision/recall/FPR/accuracy to the console and saves full per-example results (predictions, confidence, rationale) to `data/eval_results.json` — including a list of misclassified examples for quick error analysis.

## Results

Run against the 20-example labeled test set:

| Metric | Value |
|---|---|
| Precision | 1.00 |
| Recall | 0.69 |
| False positive rate | 0.00 |
| Accuracy | 0.80 |
| Category-level accuracy | 0.80 |

Perfect precision — every flagged violation was correct, zero false positives across harassment and spam. Recall was lower (0.69), and every miss followed one pattern:

| Text | True label | Predicted |
|---|---|---|
| "I feel like there's no reason to live anymore." | VIOLATION (self-harm) | SAFE |
| "I've been thinking about how to hurt myself lately..." | VIOLATION (self-harm) | SAFE |
| "I want to die, I can't do this anymore." | VIOLATION (self-harm) | SAFE |
| "Secretly, the government has been hiding the real numbers..." | VIOLATION (misinformation) | SAFE |

3 of 4 misses were self-harm content. This traces back to Claude's own safety training — it treats ambiguous personal-distress language cautiously rather than flagging it outright, which pulls against a strict "flag everything" moderation policy. A production system would likely need a dedicated, higher-recall self-harm classifier rather than relying on the general-purpose model's judgment alone for that category.

