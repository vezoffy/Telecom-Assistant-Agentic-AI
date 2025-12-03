from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from telecom_assistant.config.config import Config
from telecom_assistant.utils.database import get_database
from sqlalchemy import text
import os

# Set API Key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

# --- Tools ---

@tool
def get_customer_details(customer_id: str):
    """Get details of a customer by their ID."""
    db = get_database()
    with db._engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM customers WHERE customer_id = '{customer_id}'")).fetchone()
        if result:
            # Convert row to dict using _mapping
            return dict(result._mapping)
        return "Customer not found."

@tool
def update_customer_address(customer_id: str, new_address: str):
    """Update the address of a customer."""
    db = get_database()
    with db._engine.connect() as conn:
        # Check if customer exists
        check = conn.execute(text(f"SELECT count(*) FROM customers WHERE customer_id = '{customer_id}'")).fetchone()
        if check[0] == 0:
            return "Customer not found."
            
        conn.execute(text(f"UPDATE customers SET address = '{new_address}' WHERE customer_id = '{customer_id}'"))
        conn.commit()
        return f"Address updated successfully for {customer_id}."

@tool
def update_customer_email(customer_id: str, new_email: str):
    """Update the email of a customer."""
    db = get_database()
    with db._engine.connect() as conn:
        check = conn.execute(text(f"SELECT count(*) FROM customers WHERE customer_id = '{customer_id}'")).fetchone()
        if check[0] == 0:
            return "Customer not found."
            
        conn.execute(text(f"UPDATE customers SET email = '{new_email}' WHERE customer_id = '{customer_id}'"))
        conn.commit()
        return f"Email updated successfully for {customer_id}."

@tool
def update_customer_phone(customer_id: str, new_phone: str):
    """Update the phone number of a customer."""
    db = get_database()
    with db._engine.connect() as conn:
        check = conn.execute(text(f"SELECT count(*) FROM customers WHERE customer_id = '{customer_id}'")).fetchone()
        if check[0] == 0:
            return "Customer not found."
            
        conn.execute(text(f"UPDATE customers SET phone_number = '{new_phone}' WHERE customer_id = '{customer_id}'"))
        conn.commit()
        return f"Phone number updated successfully for {customer_id}."

@tool
def register_new_customer(name: str, email: str, phone: str, address: str, plan_id: str = "STD_500"):
    """Register a new customer."""
    db = get_database()
    with db._engine.connect() as conn:
        # Generate ID (Simple increment logic or random for now)
        # For simplicity, let's just count and add 1, or use random. 
        # Let's use a simple random suffix
        import random
        new_id = f"CUST{random.randint(1000, 9999)}"
        
        conn.execute(text(f"""
            INSERT INTO customers (customer_id, name, email, phone_number, address, service_plan_id, account_status, registration_date)
            VALUES ('{new_id}', '{name}', '{email}', '{phone}', '{address}', '{plan_id}', 'Active', DATE('now'))
        """))
        conn.commit()
        return f"Customer registered successfully with ID: {new_id}"

@tool
def update_usage_charges(usage_id: str, additional_charges: float):
    """Update additional charges for a usage record and recalculate total bill."""
    db = get_database()
    with db._engine.connect() as conn:
        # 1. Fetch current details
        row = conn.execute(text(f"SELECT total_bill_amount, additional_charges FROM customer_usage WHERE usage_id = '{usage_id}'")).fetchone()
        
        if not row:
            return f"Usage record {usage_id} not found."
            
        current_total = row[0]
        current_additional = row[1]
        
        # 2. Calculate new total
        # Base bill = Total - Old Additional
        base_bill = current_total - current_additional
        new_total = base_bill + additional_charges
        
        # 3. Update DB
        conn.execute(text(f"""
            UPDATE customer_usage 
            SET additional_charges = {additional_charges}, total_bill_amount = {new_total}
            WHERE usage_id = '{usage_id}'
        """))
        conn.commit()
        
        return f"Updated charges for {usage_id}. New Additional Charges: {additional_charges}, New Total Bill: {new_total}"

# --- Agent ---

def create_customer_management_agent():
    """Create and return a LangGraph agent for customer management"""
    
    llm = ChatOpenAI(model=Config.OPENAI_MODEL_NAME, temperature=0)
    
    tools = [
        get_customer_details,
        update_customer_address,
        update_customer_email,
        update_customer_phone,
        update_customer_phone,
        register_new_customer,
        update_usage_charges
    ]
    
    system_message = """You are a Customer Management Agent.
    You can view and update customer details, customer_usage details directly in the database.
    
    Database Schema:
    - customers: customer_id (PK), name, email, phone_number, address, service_plan_id (FK), account_status, registration_date
    - service_plans: plan_id (PK), name, monthly_cost, data_limit_gb, voice_minutes, sms_count, description
    - service_plans: plan_id (PK), name, monthly_cost, data_limit_gb, voice_minutes, sms_count, description
    - customer_usage: usage_id (PK), customer_id (FK), billing_period_end, data_used_gb, total_bill_amount, additional_charges

    Note: 'additional_charges' refers to charges for Value Added Services (VAS).
    
    Relationships:
    - `customers.service_plan_id` links to `service_plans.plan_id`.
    
    When a user asks to update their information (address, email, phone), use the appropriate tool.
    Always confirm the action was successful.

    You can also update 'additional_charges' in the `customer_usage` table using `update_usage_charges`.
    This tool automatically recalculates the `total_bill_amount`.
    
    If you need the customer ID and it's not provided in the query, ask for it (or check the context if available).
    
    If a user asks about their "current plan", use `get_customer_details` to find their `service_plan_id`, 
    and then you can explain they are on that plan (or you might need to infer details if you can't query plans directly, 
    but for now just providing the plan ID/Name from customer details is a good start).
    """
    
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message
    )
    
    return agent_executor

def process_customer_management_query(query: str, customer_id: str = None):
    """Process a customer management query"""
    
    agent_executor = create_customer_management_agent()
    
    # If customer_id is provided in context but not in query, we might want to inject it
    # But for now, let's rely on the agent asking or the user providing it, 
    # OR we append it to the query to help the agent.
    
    final_query = query
    if customer_id and customer_id not in query:
        final_query = f"{query} (Customer ID: {customer_id})"
    
    try:
        response = agent_executor.invoke({"messages": [("user", final_query)]})
        return response["messages"][-1].content
    except Exception as e:
        return f"Error processing customer management query: {e}"
