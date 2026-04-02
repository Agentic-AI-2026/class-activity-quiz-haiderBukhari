# LangGraph Planner-Executor Agent

This repository contains the converted LangGraph Planner-Executor agent, replacing an older LangChain-based structure inside `Plan-Execu.py`. 

## What We Did and How It Works

We converted a static Planner-Executor pipeline into a fully functional **State Graph** using `langgraph`. 
Here is how we mapped it:

1. **State Definition (`AgentState`)**: We defined the shared clipboard/state using Python's `TypedDict` containing four specific fields: `goal` (str), `plan` (List), `current_step` (int), and `results` (List).
2. **Planner Node (`planner_node`)**: Receives the initial `goal`, sends it to our AI (Gemini 2.5 Flash), and asks it to output a JSON structure defining specific steps, what tools to use, and how to execute it. This is saved directly to `state["plan"]`.
3. **Executor Node (`executor_node`)**: Uses the `current_step` tracker to run tasks one by one. If it identifies a tool (like `get_weather`), it triggers the corresponding function from our `tools_map` and appends the outcome to `state["results"]`. If no tool is requested, it synthesizes its own answer using the AI.
4. **Conditional Edges**: We created an edge that checks if `current_step` exceeds the total steps in the plan. If true, the agent wraps up cleanly (`END`). Otherwise, it loops right back into the `executor_node`.

## Structure

* **`graph.py`**: Contains the state definition, planner and executor nodes, conditional edge routing, and graph construction according to LangGraph workflow standards.
* **`main.py`**: The main driver program that connects to `gemini-2.5-flash` using a `.env` file, supplies our mock tools (`get_weather`, `fetch_wikipedia`, `fetch_data_source`), and drives the graph invocation. 
* **`MCP_code.py`**: A helper code block module to interface with local or remote MCP clients (we intentionally removed defunct `math` and `data` MCP bindings so the system defaults smoothly to mock tools!).
* **`Plan-Execu.py`**: The previous legacy LangChain-based reference code for the agent.

## Execution Example
When running `python main.py`, the agent successfully parses out the prompt: "Plan an outdoor event for 150 people..." into distinct tasks (i.e. calculate tables, check weather). It natively mocks "Sunny and 75F for London/New York," runs calculations mathematically, and finalizes a generated summary—all executing autonomously without user intervention!

## Running the Code

Install dependencies:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # (or directly via pip install python-dotenv langchain-google-genai langgraph langchain-core)
```

To run the file:
```bash
python main.py
```
