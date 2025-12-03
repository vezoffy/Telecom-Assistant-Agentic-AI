import streamlit as st
import os
import shutil
from telecom_assistant.config.config import Config
from telecom_assistant.utils.document_loader import load_documents
from telecom_assistant.orchestration.graph import run_orchestrator

from telecom_assistant.utils.database import get_database
from sqlalchemy import text

def get_customer_info(customer_id):
    """Fetch customer, plan, and latest usage info from DB."""
    db = get_database()
    info = {}
    
    with db._engine.connect() as conn:
        # 1. Customer Details
        cust_query = text(f"SELECT * FROM customers WHERE customer_id = '{customer_id}'")
        cust_res = conn.execute(cust_query).fetchone()
        if cust_res:
            info['name'] = cust_res[1]
            info['email'] = cust_res[2]
            info['phone'] = cust_res[3]
            plan_id = cust_res[5]
            
            # 2. Plan Details
            plan_query = text(f"SELECT * FROM service_plans WHERE plan_id = '{plan_id}'")
            plan_res = conn.execute(plan_query).fetchone()
            if plan_res:
                info['plan_name'] = plan_res[1]
                info['plan_cost'] = plan_res[2]
                info['data_limit'] = "Unlimited" if plan_res[4] else f"{plan_res[3]} GB"
            
            # 3. Latest Usage
            usage_query = text(f"SELECT * FROM customer_usage WHERE customer_id = '{customer_id}' ORDER BY billing_period_end DESC LIMIT 1")
            usage_res = conn.execute(usage_query).fetchone()
            if usage_res:
                info['data_used'] = f"{usage_res[4]} GB"
                info['bill_amount'] = f"${usage_res[8]}"
                
    return info

def render_login():
    """Renders the login page."""
    st.title("Telecom Assistant Login")
    
    with st.form("login_form"):
        username = st.text_input("Username / Customer ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username == "admin" and password == "admin":
                st.session_state["logged_in"] = True
                st.session_state["role"] = "admin"
                st.success("Logged in as Admin")
                st.rerun()
            elif password == "user":
                # Validate Customer ID in DB
                db = get_database()
                with db._engine.connect() as conn:
                    result = conn.execute(text(f"SELECT count(*) FROM customers WHERE customer_id = '{username}'")).fetchone()
                    if result[0] > 0:
                        st.session_state["logged_in"] = True
                        st.session_state["role"] = "customer"
                        st.session_state["customer_id"] = username
                        st.success(f"Logged in as Customer ({username})")
                        st.rerun()
                    else:
                        st.error("Invalid Customer ID")
            else:
                st.error("Invalid credentials")

def render_admin_dashboard():
    """Renders the Admin Dashboard for document management."""
    st.title("Admin Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Knowledge Base", "Analytics", "Admin Chat"])
    
    with tab1:
        st.header("Knowledge Base Management")
        st.write("### Upload Technical Documents")
        st.write("Supported formats: PDF, Markdown, Text")
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            accept_multiple_files=True,
            type=['pdf', 'md', 'txt']
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                docs_dir = os.path.join(Config.DATA_DIR, "documents")
                os.makedirs(docs_dir, exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    file_path = os.path.join(docs_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Updating Knowledge Base (Indexing)...")
                
                try:
                    # Reload documents to update vector store
                    load_documents()
                    st.success(f"Successfully processed {len(uploaded_files)} documents and updated the Knowledge Base!")
                except Exception as e:
                    st.error(f"Error updating knowledge base: {e}")
            else:
                st.warning("Please upload at least one file.")

    with tab2:
        st.header("System Analytics")
        import pandas as pd
        import plotly.express as px
        
        db = get_database()
        try:
            with db._engine.connect() as conn:
                # Fetch logs
                logs_df = pd.read_sql("SELECT * FROM query_logs", conn)
                
                if not logs_df.empty:
                    # 1. Key Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Queries", len(logs_df))
                    col2.metric("Avg Sentiment", f"{logs_df['sentiment_score'].mean():.2f}")
                    col3.metric("Active Users", logs_df['customer_id'].nunique())
                    
                    # 2. Category Distribution
                    st.subheader("Query Categories")
                    fig_cat = px.pie(logs_df, names='category', title='Distribution of Query Types')
                    st.plotly_chart(fig_cat)
                    
                    # 3. Sentiment Analysis
                    st.subheader("Sentiment Analysis")
                    logs_df['sentiment_label'] = logs_df['sentiment_score'].apply(
                        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
                    )
                    fig_sent = px.bar(logs_df, x='sentiment_label', title='User Sentiment', color='sentiment_label')
                    st.plotly_chart(fig_sent)
                    
                    # 4. Recent Logs
                    st.subheader("Recent Logs")
                    st.dataframe(logs_df.sort_values(by='timestamp', ascending=False).head(10))
                else:
                    st.info("No query logs available yet.")
                    
        except Exception as e:
            st.error(f"Error loading analytics: {e}")

    with tab3:
        st.header("Admin Chat (CRUD Operations)")
        st.info("Use this chat to perform administrative tasks like registering customers, updating details, or querying the database.")
        
        # Initialize admin chat history
        if "admin_messages" not in st.session_state:
            st.session_state["admin_messages"] = []
            st.session_state["admin_messages"].append({
                "role": "assistant",
                "content": "Hello Admin. I am ready to assist with customer management and other tasks."
            })
            
        if "admin_thread_id" not in st.session_state:
            import uuid
            st.session_state["admin_thread_id"] = str(uuid.uuid4())

        # Display chat messages
        for message in st.session_state["admin_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Enter admin command..."):
            # Add user message to chat history
            st.session_state["admin_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Processing admin command..."):
                    try:
                        # Run orchestrator with ADMIN ID
                        response = run_orchestrator(prompt, customer_id="ADMIN", thread_id=st.session_state["admin_thread_id"])
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state["admin_messages"].append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["messages"] = []
        st.session_state["admin_messages"] = []
        st.session_state["customer_id"] = None
        st.session_state["role"] = None
        st.rerun()

def render_customer_dashboard():
    """Renders the Customer Dashboard for chat."""
    st.title("Telecom Customer Support")
    
    customer_id = st.session_state.get("customer_id", "CUST001")
    
    # Sidebar for context
    with st.sidebar:
        st.header("Customer Profile")
        
        # Fetch and display info
        try:
            info = get_customer_info(customer_id)
            if info:
                st.subheader(info.get('name', 'Unknown'))
                st.caption(f"ID: {customer_id}")
                st.text(f"Plan: {info.get('plan_name', 'N/A')}")
                st.text(f"Limit: {info.get('data_limit', 'N/A')}")
                st.markdown("---")
                st.write("**Current Usage:**")
                st.metric("Data Used", info.get('data_used', '0 GB'))
                st.metric("Last Bill", info.get('bill_amount', '$0'))
            else:
                st.warning("Could not load profile.")
        except Exception as e:
            st.error(f"Error loading profile: {e}")
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state["messages"] = []
            st.rerun()
            
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["messages"] = []
            st.session_state["customer_id"] = None
            st.session_state["role"] = None
            st.rerun()
    
    # Initialize chat history and thread_id
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # Add welcome message
        st.session_state["messages"].append({
            "role": "assistant", 
            "content": f"Hello {info.get('name', '')}! I'm your Telecom Assistant. How can I help you today?"
        })
    
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Display chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your query here..."):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Pass thread_id instead of history
                    response = run_orchestrator(prompt, customer_id, thread_id=st.session_state["thread_id"])
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
