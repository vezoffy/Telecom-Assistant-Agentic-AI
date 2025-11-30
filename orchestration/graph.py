from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from telecom_assistant.config.config import Config
from telecom_assistant.agents.billing_agents import process_billing_query
from telecom_assistant.agents.network_agents import process_network_query
from telecom_assistant.agents.service_agents import process_recommendation_query
from telecom_assistant.agents.knowledge_agents import process_knowledge_query
import os

# Set API Key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

from telecom_assistant.orchestration.state import AgentState

# --- Nodes ---

def classify_query(state: AgentState) -> AgentState:
    """Classifies the query into one of the defined categories."""
    query = state["query"]
    
    llm = ChatOpenAI(model=Config.OPENAI_MODEL_NAME, temperature=0)
    
    prompt = PromptTemplate.from_template(
        """You are a query classifier for a telecom assistant.
        Classify the following query into exactly one of these categories:
        - BILLING: Questions about bills, charges, payments, or account balance.
        - NETWORK: Questions about signal, internet issues, outages, or device connectivity.
        - SERVICE: Questions about plan recommendations, upgrading, or new services.
        - KNOWLEDGE: General technical questions, "how-to" guides, or factual coverage/compatibility checks.
        - OTHER: Anything else.
        
        Query: {query}
        
        Category:"""
    )
    
    chain = prompt | llm
    result = chain.invoke({"query": query})
    category = result.content.strip().upper()
    
    # Normalize category just in case
    if "BILLING" in category: category = "BILLING"
    elif "NETWORK" in category: category = "NETWORK"
    elif "SERVICE" in category: category = "SERVICE"
    elif "KNOWLEDGE" in category: category = "KNOWLEDGE"
    else: category = "OTHER"
    
    print(f"--- Classified Query as: {category} ---")
    return {"category": category}

def crew_ai_node(state: AgentState) -> AgentState:
    """Handles billing queries using CrewAI."""
    print("--- Routing to Billing Agents (CrewAI) ---")
    query = state["query"]
    customer_id = state.get("customer_id", "CUST001") # Default if not provided
    
    try:
        # process_billing_query returns a CrewOutput object or string
        # We need to convert it to string
        result = process_billing_query(customer_id, query)
        return {"response": str(result)}
    except Exception as e:
        return {"response": f"Error in Billing Agent: {str(e)}"}

def autogen_node(state: AgentState) -> AgentState:
    """Handles network queries using AutoGen."""
    print("--- Routing to Network Agents (AutoGen) ---")
    query = state["query"]
    
    try:
        # process_network_query prints to stdout and returns a status string
        # For this demo, we might not capture the full conversation easily without redirecting stdout
        # or modifying the agent to return the chat history.
        # The current implementation returns "Chat completed."
        # We will run it and return the status, acknowledging the limitation.
        result = process_network_query(query)
        return {"response": f"Network Troubleshooting Session Completed. Status: {result}"}
    except Exception as e:
        return {"response": f"Error in Network Agent: {str(e)}"}

def langchain_node(state: AgentState) -> AgentState:
    """Handles service recommendations using LangChain."""
    print("--- Routing to Service Agents (LangChain) ---")
    query = state["query"]
    
    try:
        result = process_recommendation_query(query)
        return {"response": str(result)}
    except Exception as e:
        return {"response": f"Error in Service Agent: {str(e)}"}

def llamaindex_node(state: AgentState) -> AgentState:
    """Handles knowledge queries using LlamaIndex."""
    print("--- Routing to Knowledge Agents (LlamaIndex) ---")
    query = state["query"]
    
    try:
        result = process_knowledge_query(query)
        return {"response": str(result)}
    except Exception as e:
        return {"response": f"Error in Knowledge Agent: {str(e)}"}

def fallback_handler(state: AgentState) -> AgentState:
    """Handles unclassified or other queries."""
    print("--- Routing to Fallback Handler ---")
    return {"response": "I'm sorry, I couldn't understand your request. Please ask about billing, network issues, service plans, or technical support."}

# --- Routing Logic ---

def route_query(state: AgentState) -> Literal["crew_ai_node", "autogen_node", "langchain_node", "llamaindex_node", "fallback_handler"]:
    category = state["category"]
    
    if category == "BILLING":
        return "crew_ai_node"
    elif category == "NETWORK":
        return "autogen_node"
    elif category == "SERVICE":
        return "langchain_node"
    elif category == "KNOWLEDGE":
        return "llamaindex_node"
    else:
        return "fallback_handler"

# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("classify_query", classify_query)
workflow.add_node("crew_ai_node", crew_ai_node)
workflow.add_node("autogen_node", autogen_node)
workflow.add_node("langchain_node", langchain_node)
workflow.add_node("llamaindex_node", llamaindex_node)
workflow.add_node("fallback_handler", fallback_handler)

# Set Entry Point
workflow.set_entry_point("classify_query")

# Add Conditional Edges
workflow.add_conditional_edges(
    "classify_query",
    route_query,
    {
        "crew_ai_node": "crew_ai_node",
        "autogen_node": "autogen_node",
        "langchain_node": "langchain_node",
        "llamaindex_node": "llamaindex_node",
        "fallback_handler": "fallback_handler"
    }
)

# Add Edges to End
workflow.add_edge("crew_ai_node", END)
workflow.add_edge("autogen_node", END)
workflow.add_edge("langchain_node", END)
workflow.add_edge("llamaindex_node", END)
workflow.add_edge("fallback_handler", END)

# Compile Graph
app = workflow.compile()

def run_orchestrator(query: str, customer_id: str = "CUST001"):
    """Run the orchestration graph for a given query."""
    inputs = {"query": query, "customer_id": customer_id, "history": []}
    result = app.invoke(inputs)
    return result["response"]