from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from telecom_assistant.config.config import Config
from telecom_assistant.agents.billing_agents import process_billing_query
from telecom_assistant.agents.network_agents import process_network_query
from telecom_assistant.agents.service_agents import process_recommendation_query
from telecom_assistant.agents.knowledge_agents import process_knowledge_query
from telecom_assistant.agents.customer_management_agent import process_customer_management_query
from telecom_assistant.utils.database import get_database
from telecom_assistant.orchestration.state import AgentState
from textblob import TextBlob
import os
import uuid
from sqlalchemy import text

# Set API Key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

# --- Helper Functions ---

def log_query_to_db(customer_id: str, query: str, category: str):
    """Logs the query to the database with sentiment analysis."""
    try:
        blob = TextBlob(query)
        sentiment = blob.sentiment.polarity
        
        db = get_database()
        with db._engine.connect() as conn:
            stmt = text("INSERT INTO query_logs (customer_id, query_text, category, sentiment_score) VALUES (:cid, :q, :cat, :sent)")
            conn.execute(stmt, {"cid": customer_id, "q": query, "cat": category, "sent": sentiment})
            conn.commit()
            
        print(f"Logged query: {category} (Sentiment: {sentiment:.2f})")
    except Exception as e:
        print(f"Failed to log query: {e}")

def _format_query_with_history(query: str, history: list) -> str:
    """Helper to append history to query."""
    if not history:
        return query
        
    recent = history[-10:] # Take last 10 messages for context
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
    return f"Context from previous chat:\n{history_str}\n\nCurrent Query: {query}"

# --- Nodes ---

def classify_query(state: AgentState) -> AgentState:
    """Classifies the query into one of the defined categories."""
    query = state["query"]
    history = state.get("history", [])
    customer_id = state.get("customer_id", "UNKNOWN")
    
    # Check for empty query
    if not query or not query.strip():
        print("--- Empty Query Detected: Routing to Fallback ---")
        return {"category": "OTHER"}
    
    # Format history for prompt
    history_str = ""
    if history:
        # Take last 5 exchanges (10 messages)
        recent = history[-10:]
        history_str = "\nChat History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
    
    llm = ChatOpenAI(model=Config.OPENAI_MODEL_NAME, temperature=0)
    
    prompt = PromptTemplate.from_template(
        """You are a query classifier for a telecom assistant.
        Classify the following query into exactly one of these categories:
        - BILLING: Questions about bills, charges, payments, or account balance.
        - NETWORK: Questions about signal strength, network quality, internet speed, outages, or troubleshooting connectivity in specific locations.
        - SERVICE: Questions about plan recommendations, upgrading plans, or purchasing new services.
        - KNOWLEDGE: General technical questions, "how-to" guides (e.g., "how to activate roaming"), factual coverage checks, or compatibility questions. NOTE: Questions about "coverage quality" or "signal strength" should go to NETWORK.
        - CUSTOMER_MANAGEMENT: Requests to view or update personal info (e.g., "what is my name", "update address", "check my email").
        - OTHER: Anything else.
        
        {history}
        
        Query: {query}
        
        Category:"""
    )
    
    chain = prompt | llm
    result = chain.invoke({"query": query, "history": history_str})
    category = result.content.strip().upper()
    
    # Normalize category
    if "BILLING" in category: category = "BILLING"
    elif "NETWORK" in category: category = "NETWORK"
    elif "SERVICE" in category: category = "SERVICE"
    elif "KNOWLEDGE" in category: category = "KNOWLEDGE"
    elif "CUSTOMER" in category or "MANAGEMENT" in category: category = "CUSTOMER_MANAGEMENT"
    else: category = "OTHER"
    
    print(f"--- Classified Query as: {category} ---")
    
    # Log the query
    log_query_to_db(customer_id, query, category)
    
    return {"category": category}

def crew_ai_node(state: AgentState) -> AgentState:
    """Handles billing queries using CrewAI."""
    print("--- Routing to Billing Agents (CrewAI) ---")
    query = state["query"]
    history = state.get("history", [])
    customer_id = state.get("customer_id", "CUST001")
    
    # Inject history
    final_query = _format_query_with_history(query, history)
    
    try:
        result = process_billing_query(customer_id, final_query)
        response = str(result)
        return {"response": response, "history": [{"role": "assistant", "content": response}]}
    except Exception as e:
        error_msg = f"Error in Billing Agent: {str(e)}"
        return {"response": error_msg, "history": [{"role": "assistant", "content": error_msg}]}

def autogen_node(state: AgentState) -> AgentState:
    """Handles network queries using AutoGen."""
    print("--- Routing to Network Agents (AutoGen) ---")
    query = state["query"]
    history = state.get("history", [])
    
    final_query = _format_query_with_history(query, history)
    
    try:
        result = process_network_query(final_query)
        response = f"Network Troubleshooting Session Completed. Status: {result}"
        return {"response": response, "history": [{"role": "assistant", "content": response}]}
    except Exception as e:
        error_msg = f"Error in Network Agent: {str(e)}"
        return {"response": error_msg, "history": [{"role": "assistant", "content": error_msg}]}

def langchain_node(state: AgentState) -> AgentState:
    """Handles service recommendations using LangChain."""
    print("--- Routing to Service Agents (LangChain) ---")
    query = state["query"]
    history = state.get("history", [])
    
    final_query = _format_query_with_history(query, history)
    
    try:
        result = process_recommendation_query(final_query)
        response = str(result)
        return {"response": response, "history": [{"role": "assistant", "content": response}]}
    except Exception as e:
        error_msg = f"Error in Service Agent: {str(e)}"
        return {"response": error_msg, "history": [{"role": "assistant", "content": error_msg}]}

def llamaindex_node(state: AgentState) -> AgentState:
    """Handles knowledge queries using LlamaIndex."""
    print("--- Routing to Knowledge Agents (LlamaIndex) ---")
    query = state["query"]
    history = state.get("history", [])
    
    final_query = _format_query_with_history(query, history)
    
    try:
        result = process_knowledge_query(final_query)
        response = str(result)
        return {"response": response, "history": [{"role": "assistant", "content": response}]}
    except Exception as e:
        error_msg = f"Error in Knowledge Agent: {str(e)}"
        return {"response": error_msg, "history": [{"role": "assistant", "content": error_msg}]}

def customer_management_node(state: AgentState) -> AgentState:
    """Handles customer management queries."""
    print("--- Routing to Customer Management Agent ---")
    query = state["query"]
    history = state.get("history", [])
    customer_id = state.get("customer_id")
    
    final_query = _format_query_with_history(query, history)
    
    try:
        result = process_customer_management_query(final_query, customer_id)
        response = str(result)
        return {"response": response, "history": [{"role": "assistant", "content": response}]}
    except Exception as e:
        error_msg = f"Error in Customer Management Agent: {str(e)}"
        return {"response": error_msg, "history": [{"role": "assistant", "content": error_msg}]}

def fallback_handler(state: AgentState) -> AgentState:
    """Handles unclassified or other queries."""
    print("--- Routing to Fallback Handler ---")
    response = "I'm sorry, I couldn't understand your request. Please ask about billing, network issues, service plans, or technical support."
    return {"response": response, "history": [{"role": "assistant", "content": response}]}

# --- Routing Logic ---

def route_query(state: AgentState) -> Literal["crew_ai_node", "autogen_node", "langchain_node", "llamaindex_node", "customer_management_node", "fallback_handler"]:
    category = state["category"]
    
    if category == "BILLING":
        return "crew_ai_node"
    elif category == "NETWORK":
        return "autogen_node"
    elif category == "SERVICE":
        return "langchain_node"
    elif category == "KNOWLEDGE":
        return "llamaindex_node"
    elif category == "CUSTOMER_MANAGEMENT":
        return "customer_management_node"
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
workflow.add_node("customer_management_node", customer_management_node)
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
        "customer_management_node": "customer_management_node",
        "fallback_handler": "fallback_handler"
    }
)

# Add Edges to End
workflow.add_edge("crew_ai_node", END)
workflow.add_edge("autogen_node", END)
workflow.add_edge("langchain_node", END)
workflow.add_edge("llamaindex_node", END)
workflow.add_edge("customer_management_node", END)
workflow.add_edge("fallback_handler", END)

# Initialize MemorySaver
checkpointer = MemorySaver()

# Compile Graph with Checkpointer
app = workflow.compile(checkpointer=checkpointer)

def run_orchestrator(query: str, customer_id: str = "CUST001", thread_id: str = None):
    """Run the orchestration graph for a given query."""
    if not thread_id:
        thread_id = str(uuid.uuid4())
        
    config = {"configurable": {"thread_id": thread_id}}
    
    # Add user message to history in input
    # Since 'history' has operator.add, this will append to existing history
    user_msg = {"role": "user", "content": query}
    
    inputs = {
        "query": query, 
        "customer_id": customer_id, 
        "history": [user_msg] 
    }
    
    result = app.invoke(inputs, config=config)
    return result["response"]