from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import mlflow
import json
from typing import TypedDict

# ── State shared across all nodes ──────────────────────────────────────────
class PipelineState(TypedDict):
    run_id: str
    metrics: dict
    analysis: str
    decision: str
    reason: str

# ── Node 1: Evaluate ───────────────────────────────────────────────────────
def evaluate_node(state: PipelineState) -> PipelineState:
    print("\n[evaluate_node] Fetching metrics from MLflow...")
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(state["run_id"])
    metrics = run.data.metrics

    relevant = {
        "test_accuracy": round(metrics.get("test_accuracy", 0), 4),
        "test_f1":       round(metrics.get("test_f1", 0), 4),
        "test_auc":      round(metrics.get("test_auc", 0), 4),
    }

    print(f"[evaluate_node] Metrics: {relevant}")
    return {**state, "metrics": relevant}

# ── Node 2: Gate (LLM reasons about deployment) ────────────────────────────
def gate_node(state: PipelineState) -> PipelineState:
    print("\n[gate_node] Asking LLM to evaluate model readiness...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    prompt = f"""
You are an ML deployment gatekeeper. Evaluate these metrics carefully.

Model metrics (already computed, do not recalculate):
- Accuracy: {state['metrics']['test_accuracy']} (threshold: >= 0.82, PASS: {state['metrics']['test_accuracy'] >= 0.82})
- F1 Score: {state['metrics']['test_f1']} (threshold: >= 0.60, PASS: {state['metrics']['test_f1'] >= 0.60})
- AUC:      {state['metrics']['test_auc']} (threshold: >= 0.85, PASS: {state['metrics']['test_auc'] >= 0.85})

If ALL three show PASS: True, decision must be APPROVE. Otherwise REJECT.

Respond ONLY with a valid JSON object:
{{"decision": "APPROVE" or "REJECT", "reason": "one sentence explanation"}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"[gate_node] LLM response: {response.content}")

    try:
        # Clean response in case LLM adds extra text
        content = response.content.strip()
        start = content.find("{")
        end   = content.rfind("}") + 1
        parsed = json.loads(content[start:end])
        decision = parsed.get("decision", "REJECT")
        reason   = parsed.get("reason", "No reason provided")
    except Exception as e:
        print(f"[gate_node] Parse error: {e} — defaulting to REJECT")
        decision = "REJECT"
        reason   = "Failed to parse LLM response"

    return {**state, "decision": decision, "reason": reason}

# ── Node 3: Decision ───────────────────────────────────────────────────────
def decision_node(state: PipelineState) -> PipelineState:
    print(f"\n[decision_node] Final decision: {state['decision']}")
    print(f"[decision_node] Reason: {state['reason']}")
    return state

# ── Routing function ───────────────────────────────────────────────────────
def route_after_gate(state: PipelineState) -> str:
    return "decision_node"

# ── Build the graph ────────────────────────────────────────────────────────
def build_agent_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("evaluate_node", evaluate_node)
    graph.add_node("gate_node",     gate_node)
    graph.add_node("decision_node", decision_node)

    graph.set_entry_point("evaluate_node")
    graph.add_edge("evaluate_node", "gate_node")
    graph.add_edge("gate_node",     "decision_node")
    graph.add_edge("decision_node", END)

    return graph.compile()

# ── Run ────────────────────────────────────────────────────────────────────
def run_agent_pipeline(run_id: str) -> dict:
    pipeline = build_agent_pipeline()
    initial_state = PipelineState(
        run_id=run_id,
        metrics={},
        analysis="",
        decision="",
        reason=""
    )
    final_state = pipeline.invoke(initial_state)
    return {
        "decision": final_state["decision"],
        "reason":   final_state["reason"],
        "metrics":  final_state["metrics"]
    }

if __name__ == "__main__":
    with open("models/latest_run_id.txt") as f:
        run_id = f.read().strip()
    result = run_agent_pipeline(run_id)
    print(f"\n{'='*50}")
    print(f"AGENT VERDICT: {result['decision']}")
    print(f"REASON: {result['reason']}")
    print(f"METRICS: {result['metrics']}")
    print(f"{'='*50}")