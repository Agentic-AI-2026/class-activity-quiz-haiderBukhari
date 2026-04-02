import asyncio
import os
from dotenv import load_dotenv
from graph import create_graph

# Load environment variables from .env
load_dotenv()

# Setup LLM to strictly use Google Generative AI with the API key from .env
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
except ImportError:
    print("Please make sure you have langchain_google_genai installed: pip install langchain-google-genai")
    exit(1)

async def main():
    print("Initializing LangGraph Planner-Executor Agent...\n")
    
    # Try loading real MCP tools if standard setup is ok
    try:
        from MCP_code import get_mcp_tools
        tools, tools_map = await get_mcp_tools(["weather", "search"])
    except Exception as e:
        print(f"Failed to load MCP tools: {e}. Falling back to mock tools.")
        # Fallback to mock tools to demonstrate execution flow
        tools_map = {
            "fetch_wikipedia": lambda args: f"Mock Wikipedia content for {args.get('topic')}",
            "fetch_data_source": lambda args: f"Mock Data results for {args.get('source')}",
            "get_weather": lambda args: f"Mock weather: Sunny and 75F for {args.get('city')}",
        }

    # Configuration for our graph nodes
    config = {
        "configurable": {
            "llm": llm,
            "tools_map": tools_map
        }
    }
    
    app = create_graph()
    
    goal = "Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize."
    
    print("=" * 60)
    # We execute the graph asynchronously
    final_state = await app.ainvoke(
        {"goal": goal},
        config=config
    )
    
    print("=" * 60)
    print("Execution Finished! Final Expected Results:")
    for res in final_state.get("results", []):
        print(f"Step {res['step']}:\n{res['result']}\n")

if __name__ == "__main__":
    asyncio.run(main())
