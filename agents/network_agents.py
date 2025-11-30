import autogen
from telecom_assistant.config.config import Config
from telecom_assistant.utils.database import get_database
from telecom_assistant.utils.document_loader import load_documents
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os

# Set API Key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

def create_network_agents():
    """Create and return an AutoGen group chat for network troubleshooting"""
    
    # Configuration for agents
    config_list = [{
        "model": Config.OPENAI_MODEL_NAME,
        "api_key": Config.OPENAI_API_KEY,
    }]
    
    llm_config = {
        "config_list": config_list,
        "temperature": 0.2,
        "timeout": 120,
    }

    # --- Tools Setup ---
    
    # 1. Network Status Tool
    def check_network_status(location: str) -> str:
        """Check for network outages or incidents in a specific location."""
        db = get_database()
        # Simple SQL query to check status
        # Note: In a real app, use parameterized queries or an ORM. 
        # For this hackathon/demo with trusted input, f-string is acceptable but risky.
        # Let's use the SQL tool for safety if possible, or just raw execution for simplicity here.
        from sqlalchemy import text
        with db._engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM network_status WHERE location LIKE '%{location}%'")).fetchall()
            if not result:
                return f"No reported network incidents found in {location}."
            return str(result)

    # 2. Troubleshooting Docs Tool
    def search_troubleshooting_docs(query: str) -> str:
        """Search technical documentation for troubleshooting steps."""
        try:
            index = load_documents()
            if not index:
                return "Error: Document index not available."
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            return f"Error searching docs: {str(e)}"

    # --- Agents Setup ---

    # 1. User Proxy Agent
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        system_message="""You represent a customer with a network issue. Your job is to:
        1. Present the customer's problem clearly.
        2. Ask clarifying questions if agents need more information.
        3. Summarize the final solution in simple terms once the agents have provided a resolution.
        4. Terminate the chat when a solution is found and summarized.""",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
    )

    # 2. Network Diagnostics Agent
    network_agent = autogen.AssistantAgent(
        name="Network_Diagnostics_Agent",
        system_message="""You are a network diagnostics expert who analyzes connectivity issues.
        Your responsibilities:
        1. Check for known outages or incidents in the customer's area using `check_network_status`.
        2. Analyze network performance metrics.
        3. Identify patterns that indicate specific network problems.
        4. Determine if the issue is widespread or localized to the customer.
        
        Always begin by checking the network status database for outages in the customer's region before suggesting device-specific solutions.""",
        llm_config=llm_config,
    )

    # 3. Device Expert Agent
    device_agent = autogen.AssistantAgent(
        name="Device_Expert_Agent",
        system_message="""You are a device troubleshooting expert who knows how to resolve connectivity issues on different phones and devices.
        Your responsibilities:
        1. Suggest device-specific settings to check.
        2. Provide step-by-step instructions for configuration using `search_troubleshooting_docs` to find accurate info.
        3. Explain how to diagnose hardware vs. software issues.
        4. Recommend specific actions based on the device type.
        
        Always ask for the device model if it's not specified, as troubleshooting steps differ between iOS, Android, and other devices.""",
        llm_config=llm_config,
    )

    # 4. Solution Integrator Agent
    integrator_agent = autogen.AssistantAgent(
        name="Solution_Integrator_Agent",
        system_message="""You are a solution integrator who combines technical analysis into actionable plans for customers.
        Your responsibilities:
        1. Synthesize information from the network and device experts.
        2. Create a prioritized list of troubleshooting steps.
        3. Present solutions in order from simplest to most likely to succeed.
        4. When a clear solution plan is formed, output the final answer and append "TERMINATE" to end the conversation.""",
        llm_config=llm_config,
    )

    # Register Tools
    # We register tools with the agents that need them and the user proxy (executor)
    
    # Network Agent needs check_network_status
    autogen.register_function(
        check_network_status,
        caller=network_agent,
        executor=user_proxy,
        name="check_network_status",
        description="Check for network outages in a specific location."
    )

    # Device Agent needs search_troubleshooting_docs
    autogen.register_function(
        search_troubleshooting_docs,
        caller=device_agent,
        executor=user_proxy,
        name="search_troubleshooting_docs",
        description="Search technical docs for troubleshooting steps."
    )

    # --- Group Chat Setup ---
    
    groupchat = autogen.GroupChat(
        agents=[user_proxy, network_agent, device_agent, integrator_agent],
        messages=[],
        max_round=12
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    return user_proxy, manager

def process_network_query(query: str):
    """Process a network troubleshooting query using AutoGen agents"""
    
    user_proxy, manager = create_network_agents()
    
    # Initiate the chat
    user_proxy.initiate_chat(
        manager,
        message=query
    )
    
    # In a real app, we might want to return the chat history or the last message
    # For now, the chat prints to stdout
    return "Chat completed."

if __name__ == "__main__":
    # Test run
    print("Starting Network Agents Test...")
    try:
        process_network_query("I have no internet in New York. My phone is an iPhone 14.")
    except Exception as e:
        print(f"Error running network agents: {e}")
