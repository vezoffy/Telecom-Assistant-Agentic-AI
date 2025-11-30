import streamlit as st
import os
import shutil
from telecom_assistant.config.config import Config
from telecom_assistant.utils.document_loader import load_documents
from telecom_assistant.orchestration.graph import run_orchestrator

def render_login():
    """Renders the login page."""
    st.title("Telecom Assistant Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username == "admin" and password == "admin":
                st.session_state["logged_in"] = True
                st.session_state["role"] = "admin"
                st.success("Logged in as Admin")
                st.rerun()
            elif username == "user" and password == "user":
                st.session_state["logged_in"] = True
                st.session_state["role"] = "customer"
                st.session_state["customer_id"] = "CUST001" # Default
                st.success("Logged in as Customer")
                st.rerun()
            else:
                st.error("Invalid credentials")

def render_admin_dashboard():
    """Renders the Admin Dashboard for document management."""
    st.title("Admin Dashboard - Knowledge Base")
    
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
            
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

def render_customer_dashboard():
    """Renders the Customer Dashboard for chat."""
    st.title("Telecom Customer Support")
    
    # Sidebar for context
    with st.sidebar:
        st.header("Customer Details")
        customer_id = st.text_input("Customer ID", value=st.session_state.get("customer_id", "CUST001"))
        st.session_state["customer_id"] = customer_id
        
        if st.button("Clear Chat History"):
            st.session_state["messages"] = []
            st.rerun()
            
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # Add welcome message
        st.session_state["messages"].append({
            "role": "assistant", 
            "content": "Hello! I'm your Telecom Assistant. How can I help you today with your billing, network, or service plans?"
        })

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
                    response = run_orchestrator(prompt, customer_id)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
