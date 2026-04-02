import json
import re
import asyncio
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START

# ─── STATE DEFINITION ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    goal: str
    plan: List[Dict[str, Any]]
    current_step: int
    results: List[Dict[str, Any]]

# ─── CONSTANTS & PROMPTS ───────────────────────────────────────────────────────
PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools and their EXACT argument names:
  - fetch_wikipedia(topic: str)       → look up a topic on Wikipedia
  - fetch_data_source(source: str)    → source must be one of: sales, customers, expenses
  - get_weather(city: str)            → get real weather for a city

Use null for tool/args on synthesis or writing steps.
Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ARG_MAP = {
    "fetch_wikipedia":  "topic",
    "fetch_data_source": "source",
    "get_weather":      "city",
}

def safe_args(tool_name: str, raw_args: dict) -> dict:
    """Remap hallucinated arg names to the correct parameter."""
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    print(f"  Remapped {raw_args} → {{'{expected}': '{first_val}'}}")
    return {expected: str(first_val)}

# ─── NODES ─────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Planner Node: Takes the user goal, calls the LLM, generates a structured plan, 
    and stores it in the state.
    """
    llm = config["configurable"].get("llm")
    if not llm:
        raise ValueError("LLM not provided in configurable settings.")
        
    goal = state["goal"]
    
    print(f" Goal: {goal}\n")
    plan_resp = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=goal)
    ])
    
    raw_text = plan_resp.content if isinstance(plan_resp.content, str) else plan_resp.content[0].get("text", "")
    
    # Clean and Parse
    clean_json = re.sub(r"```json|```", "", raw_text).strip()
    try:
        plan = json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\nRaw output: {clean_json}")
        plan = []
        
    print(f" Plan ({len(plan)} steps):")
    for s in plan:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")
    print()
    
    return {"plan": plan, "current_step": 0, "results": []}

async def executor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executor Node: Executes one step at a time. Uses a tool if specified,
    otherwise uses the LLM to synthesize based on context.
    Stores the result, and advances to the next step.
    """
    llm = config["configurable"].get("llm")
    tools_map = config["configurable"].get("tools_map", {})
    
    plan = state.get("plan", [])
    current_step_idx = state.get("current_step", 0)
    results = state.get("results", [])
    
    if current_step_idx >= len(plan):
        return {}
        
    step = plan[current_step_idx]
    tool_name = step.get("tool")
    
    print(f"  Executing Step {step['step']}: {step['description']}")
    
    if tool_name and tool_name in tools_map:
        corrected = safe_args(tool_name, step.get("args") or {})
        tool_exec = tools_map[tool_name]
        
        # Depending on if tool is async or sync, await it or just call
        if hasattr(tool_exec, 'ainvoke'):
            result = await tool_exec.ainvoke(corrected)
        elif hasattr(tool_exec, 'invoke'):
            result = tool_exec.invoke(corrected)
        else:
            # Fallback for simple callable tools mock
            if asyncio.iscoroutinefunction(tool_exec):
                result = await tool_exec(corrected)
            else:
                result = tool_exec(corrected)
    else:
        # Synthesis step — use LLM with prior results as context
        context = "\n".join([f"Step {r['step']}: {r['result']}" for r in results])
        response = llm.invoke([
            HumanMessage(content=f"{step['description']}\n\nContext:\n{context}")
        ])
        result = response.content
        
    print(f"    {str(result)}\n")
    
    step_result = {"step": step["step"], "description": step["description"], "result": str(result)}
    
    return {
        "results": [*results, step_result],
        "current_step": current_step_idx + 1
    }

def should_continue(state: AgentState) -> str:
    """
    Condition to check if all steps in the plan have been executed.
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    if current_step >= len(plan):
        return "end"
    return "execute"

# ─── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    
    workflow.add_edge(START, "planner")
    
    # We execute one step per iteration 
    workflow.add_edge("planner", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "execute": "executor",
            "end": END
        }
    )
    
    return workflow.compile()
