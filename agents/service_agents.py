from langgraph.prebuilt import create_react_agent
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from telecom_assistant.utils.database import get_database
from telecom_assistant.utils.document_loader import load_documents
from telecom_assistant.config.config import Config
import os

# Set API Key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

# Define a prompt template for service recommendations
SERVICE_RECOMMENDATION_TEMPLATE = """You are a telecom service advisor who helps customers find the best plan for their needs.
When recommending plans, consider:
1. The customer's usage patterns (data, voice, SMS)
2. Number of people/devices that will use the plan
3. Special requirements (international calling, streaming, etc.)
4. Budget constraints

Always explain WHY a particular plan is a good fit for their needs.
"""

def estimate_data_usage(activities: str) -> str:
    """
    Estimate monthly data usage based on activities.
    Example input: "streaming 2 hours of video daily, browsing 3 hours, video calls 1 hour weekly"
    """
    # Simple heuristic estimation
    total_gb = 0.0
    activities = activities.lower()
    
    if "streaming" in activities:
        # Assume HD streaming ~3GB/hr
        total_gb += 3.0 * 30 * 2 # Mock: 2 hours daily
    if "browsing" in activities:
        # Assume 0.1GB/hr
        total_gb += 0.1 * 30 * 3 # Mock: 3 hours daily
    if "video call" in activities:
        # Assume 1GB/hr
        total_gb += 1.0 * 4 * 1 # Mock: 1 hour weekly
        
    # If no specific keywords found, return a generic estimate or ask for more info
    if total_gb == 0:
        return "Could not estimate usage from description. Please specify hours for streaming, browsing, etc."
        
    return f"Estimated monthly data usage: {total_gb} GB"

def search_service_docs(query: str) -> str:
    """Search service plan documentation for qualitative details (benefits, terms)."""
    try:
        index = load_documents()
        if not index:
            return "Error: Document index not available."
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error searching docs: {str(e)}"

def create_service_agent():
    """Create and return a LangGraph agent for service recommendations"""
    
    # Create LLM
    llm = ChatOpenAI(model=Config.OPENAI_MODEL_NAME, temperature=0.2)
    
    # Create Tools
    db = get_database()
    sql_tool = QuerySQLDataBaseTool(db=db)
    python_tool = PythonREPLTool()
    
    usage_tool = Tool(
        name="estimate_data_usage",
        func=estimate_data_usage,
        description="Estimate monthly data usage based on activity descriptions."
    )
    
    vector_tool = Tool(
        name="search_service_docs",
        func=search_service_docs,
        description="Search for qualitative plan details, benefits, and terms in the documentation."
    )
    
    tools = [sql_tool, python_tool, usage_tool, vector_tool]
    
    # Create Agent using LangGraph prebuilt
    # messages_modifier acts as the system message
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SERVICE_RECOMMENDATION_TEMPLATE
    )
    
    return agent_executor

def process_recommendation_query(query: str):
    """Process a service recommendation query using the LangGraph agent"""
    
    agent_executor = create_service_agent()
    
    try:
        # LangGraph invoke takes {"messages": [...]}
        response = agent_executor.invoke({"messages": [("user", query)]})
        # The last message in the state is the AI's final response
        return response["messages"][-1].content
    except Exception as e:
        return f"Error processing recommendation: {e}"

if __name__ == "__main__":
    # Test run
    print("Starting Service Agent Test...")
    print(process_recommendation_query("What is the best plan for a family of 4 with high data usage?"))
