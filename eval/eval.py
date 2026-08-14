import asyncio
import importlib.util
import json
import os
import sys

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.json")
AGENT_PATH = os.path.join(os.path.dirname(__file__), "..", "agent", "agent.py")

_spec = importlib.util.spec_from_file_location("policy_agent_module", AGENT_PATH)
_agent_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agent_module)
classify = _agent_module.classify


async def run_eval():
    with open(TEST_SET_PATH) as f:
        test_set = json.load(f)

    results = []
    for i, item in enumerate(test_set):
        print(f"[{i+1}/{len(test_set)}] classifying: {item['text'][:50]}...")
        prediction = await classify(item["text"])
        results.append({
            "text": item["text"],
            "true_label": item["true_label"],
            "true_rule_id": item["true_rule_id"],
            "predicted_label": prediction.get("decision"),
            "predicted_rule_id": prediction.get("rule_id"),
            "confidence": prediction.get("confidence"),
            "rationale": prediction.get("rationale"),
        })

    # --- Metrics ---
    # Treat "VIOLATION" as the positive class.
    tp = sum(1 for r in results if r["true_label"] == "VIOLATION" and r["predicted_label"] == "VIOLATION")
    fp = sum(1 for r in results if r["true_label"] == "SAFE" and r["predicted_label"] == "VIOLATION")
    fn = sum(1 for r in results if r["true_label"] == "VIOLATION" and r["predicted_label"] == "SAFE")
    tn = sum(1 for r in results if r["true_label"] == "SAFE" and r["predicted_label"] == "SAFE")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    # Category-level accuracy (did it get the right specific rule_id, not just VIOLATION/SAFE)
    category_correct = sum(1 for r in results if r["predicted_rule_id"] == r["true_rule_id"])
    category_accuracy = category_correct / len(results) if results else 0.0

    metrics = {
        "n_examples": len(results),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_positive_rate": round(false_positive_rate, 3),
        "accuracy": round(accuracy, 3),
        "category_level_accuracy": round(category_accuracy, 3),
    }

    output = {"metrics": metrics, "results": results}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Evaluation results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nFull results saved to {RESULTS_PATH}")

    # Print misclassifications for quick error analysis
    misclassified = [r for r in results if r["predicted_label"] != r["true_label"]]
    if misclassified:
        print(f"\n=== Misclassified ({len(misclassified)}) ===")
        for r in misclassified:
            print(f"  TEXT: {r['text'][:60]}")
            print(f"    true={r['true_label']} ({r['true_rule_id']})  predicted={r['predicted_label']} ({r['predicted_rule_id']})")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment first.")
        sys.exit(1)
    asyncio.run(run_eval())
