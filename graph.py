from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator
from langchain_ollama import ChatOllama

# 1. Define the State
class AgentState(TypedDict):
    query: str
    steps: Annotated[List[str], operator.add]
    final_report: str

# 2. Setup Local LLM (Ensure Ollama is running)
# Run 'ollama run llama3' in your terminal first
llm = ChatOllama(model="llama3")

# 3. Define Nodes
def researcher(state: AgentState):
    print(f"--- Simulating local research for: {state['query']} ---")
    # In a real local setup, you would query your local vector DB here
    mock_data = "Local context: Agentic workflows use state machines to manage tool loops."
    return {"steps": [f"Retrieved local data for: {state['query']}", mock_data]}

def summarizer(state: AgentState):
    print("--- Synthesizing report locally ---")
    prompt = f"Using this info: {state['steps']}, write a report on: {state['query']}"
    response = llm.invoke(prompt)
    return {"final_report": response.content}

# 4. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("summarizer", summarizer)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", END)

app = workflow.compile()

# 5. Execute
if __name__ == "__main__":
    result = app.invoke({"query": "What are agentic workflows?"})
    print("\n--- FINAL REPORT ---\n")
    print(result["final_report"])