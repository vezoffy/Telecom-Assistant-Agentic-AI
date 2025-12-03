import autogen
from telecom_assistant.config.config import Config
from telecom_assistant.utils.database import get_database
from telecom_assistant.utils.document_loader import load_documents
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os

# Set API Key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

def create_network_agents(customer_id: str = "CUST001"):
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

    # Fetch customer details
    db = get_database()
    customer_name = "Unknown"
    from sqlalchemy import text
    with db._engine.connect() as conn:
        result = conn.execute(text(f"SELECT name FROM customers WHERE customer_id = '{customer_id}'")).fetchone()
        if result:
            customer_name = result[0]

    # --- Tools Setup ---
    
    # 1. Network Status Tool
    def check_network_status(city: str, district: str = None) -> str:
        """Check for network outages or incidents in a specific location."""
        db = get_database()
        from sqlalchemy import text
        with db._engine.connect() as conn:
            # Search by city primarily. If district is provided, we could refine, 
            # but usually status is reported at city or region level.
            # We'll check for matches on the city name.
            query = text(f"SELECT * FROM network_status WHERE location LIKE '%{city}%'")
            result = conn.execute(query).fetchall()
            
            if not result:
                return f"No reported network incidents found in {city}."
            return str(result)

    # 2. Coverage Quality Tools
    
    def check_location_coverage(city: str, district: str = None, technology: str = "5G") -> str:
        """Check for coverage quality in a specific location (city and optional district)."""
        db = get_database()
        from sqlalchemy import text
        with db._engine.connect() as conn:
            # Construct query based on whether district is provided
            if district:
                area_query = text(f"SELECT area_id FROM service_areas WHERE city LIKE '%{city}%' AND district LIKE '%{district}%'")
                location_str = f"{city} ({district})"
            else:
                area_query = text(f"SELECT area_id FROM service_areas WHERE city LIKE '%{city}%'")
                location_str = city
                
            area_result = conn.execute(area_query).fetchone()
            
            if not area_result:
                return f"No service area found for {location_str}."
            
            area_id = area_result[0]
            
            # Now check coverage quality
            coverage_query = text(f"SELECT * FROM coverage_quality WHERE area_id = '{area_id}' AND technology = '{technology}'")
            coverage_result = conn.execute(coverage_query).fetchall()
            
            if not coverage_result:
                return f"No coverage data found for {technology} in {location_str}."
            
            return str(coverage_result)

    def check_my_coverage(technology: str = "5G") -> str:
        """Check for coverage quality in the customer's inferred location."""
        # Use customer_id from the closure
        target_customer_id = customer_id
        db = get_database()
        from sqlalchemy import text
        with db._engine.connect() as conn:
            # Infer location from customer data
            print(f"DEBUG: Inferring location for customer_id: {target_customer_id}")
            cust_query = text(f"SELECT address FROM customers WHERE customer_id = '{target_customer_id}'")
            cust_result = conn.execute(cust_query).fetchone()
            
            location = None
            if cust_result and cust_result[0]:
                address = cust_result[0]
                # Simple heuristic: assume city is one of the known cities or part of the address string
                known_cities = ['Bangalore', 'Mumbai', 'Delhi', 'New York', 'Los Angeles', 'Chicago', 'Hyderabad']
                for city in known_cities:
                    if city.lower() in address.lower():
                        location = city
                        break
            
            if not location:
                print("DEBUG: check_my_coverage could not infer location.")
                return "Could not infer your location from your profile. Please provide a specific city."

            print(f"DEBUG: check_my_coverage inferred location: {location}")
            return check_location_coverage(location, technology)



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
        system_message=f"""You represent a customer with a network issue. You are Customer {customer_id} ({customer_name}). Your job is to:
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
    network_agent_system_message = f"""You are a network diagnostics expert who analyzes connectivity issues.
        Your responsibilities:
        1. Check for known outages or incidents in the customer's area using `check_network_status`.
        2. Analyze network performance metrics.
        3. Identify patterns that indicate specific network problems.
        4. Determine if the issue is widespread or localized to the customer.
        5. Check coverage quality:
           - If the user provides a location, extract the City and District (if available).
           - Example: "Mumbai West" -> City="Mumbai", District="West".
           - Use `check_location_coverage(city="Mumbai", district="West")`.
           - If only city is known: `check_location_coverage(city="Delhi")`.
           - If the user asks about "my city" or "my location" AND does not provide a specific city, use `check_my_coverage()`.
        
        You have access to the following network infrastructure tables:
        - customers: customer_id (PK), name, phone_number, location, device_type, device_model, device_os
        - service_areas: area_id (PK), city, district, region, terrain_type
        - cell_towers: tower_id (PK), area_id (FK), latitude, longitude, tower_type, operational_status
        - tower_technologies: tower_tech_id (PK), tower_id (FK), technology (4G/5G), frequency_band, max_capacity_mbps
        - coverage_quality: coverage_id (PK), area_id (FK), technology, signal_strength_category, avg_download_speed_mbps
        - network_status: status_id (PK), location, status, incident_type, description, estimated_resolution
        - common_network_issues: issue_id (PK), issue_category, troubleshooting_steps, resolution_approach
        - device_compatibility: compatibility_id (PK), device_model, known_issues, recommended_settings

        Relationships:
        - `cell_towers.area_id` links to `service_areas.area_id`.
        - `tower_technologies.tower_id` links to `cell_towers.tower_id`.
        - `coverage_quality.area_id` links to `service_areas.area_id`.
        
        Always begin by checking the network status database for outages in the customer's region before suggesting device-specific solutions."""

    network_agent = autogen.AssistantAgent(
        name="Network_Diagnostics_Agent",
        system_message=network_agent_system_message,
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
        description="Check for network outages in a specific location (City, optional District)."
    )

    # Device Agent needs search_troubleshooting_docs
    autogen.register_function(
        search_troubleshooting_docs,
        caller=device_agent,
        executor=user_proxy,
        name="search_troubleshooting_docs",
        description="Search technical docs for troubleshooting steps."
    )

    # Network Agent needs check_location_coverage
    autogen.register_function(
        check_location_coverage,
        caller=network_agent,
        executor=user_proxy,
        name="check_location_coverage",
        description="Check for coverage quality in a specific location (City, optional District)."
    )

    # Network Agent needs check_my_coverage
    autogen.register_function(
        check_my_coverage,
        caller=network_agent,
        executor=user_proxy,
        name="check_my_coverage",
        description="Check for coverage quality in the customer's inferred location."
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

def process_network_query(query: str, customer_id: str = "CUST001"):
    """Process a network troubleshooting query using AutoGen agents"""
    
    user_proxy, manager = create_network_agents(customer_id)
    
    # Initiate the chat
    user_proxy.initiate_chat(
        manager,
        message=query
    )
    
    # Extract the response from the group chat history
    messages = manager.groupchat.messages
    
    # Look for the last message from the Solution Integrator
    for msg in reversed(messages):
        if msg.get("name") == "Solution_Integrator_Agent":
            content = msg.get("content", "")
            # Remove the termination keyword
            return content.replace("TERMINATE", "").strip()
            
    # Fallback: Return the last message if Solution Integrator didn't speak (unlikely)
    if messages:
        return messages[-1].get("content", "")
        
    return "No response generated."

if __name__ == "__main__":
    # Test run
    print("Starting Network Agents Test...")
    try:
        process_network_query("I have no internet in New York. My phone is an iPhone 14.")
    except Exception as e:
        print(f"Error running network agents: {e}")
