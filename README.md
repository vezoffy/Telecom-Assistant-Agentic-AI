# Telecom Assistant Agentic AI

## Overview

The **Telecom Assistant** is an advanced multi-agent system designed to streamline customer support for telecom services. It leverages an orchestration layer built with **LangGraph** to intelligently route user queries to specialized agents powered by different AI frameworks. This architecture ensures that each type of request—whether it's a billing inquiry, a network issue, or a general question—is handled by the most capable tool for the job.

## Key Features

- **Intelligent Query Routing**: Uses an LLM-based classifier to categorize user intent (Billing, Network, Service, Knowledge, Customer Management) and route it to the appropriate workflow.
- **Multi-Framework Orchestration**:
  - **Billing Support**: Utilizes **CrewAI** to manage complex billing queries and tasks.
  - **Network Troubleshooting**: Deploys **AutoGen** agents to simulate and resolve technical network issues.
  - **Service Recommendations**: Powered by **LangChain** for personalized plan advice.
  - **Knowledge Base**: Integrates **LlamaIndex** for Retrieval-Augmented Generation (RAG) to answer general questions.
- **Sentiment Analysis**: Automatically analyzes and logs the sentiment of user queries using `TextBlob`.
- **Interactive UI**: A user-friendly **Streamlit** interface offering distinct views for:
  - **Customers**: To ask questions and manage their accounts.
  - **Admins**: To monitor system performance and query logs.

## Architecture

The system is organized into the following modules:

- `orchestration/`: Contains the core logic for the **LangGraph** workflow and state management (`graph.py`, `state.py`).
- `agents/`: Houses the specialized agent implementations:
  - `billing_agents.py` (CrewAI)
  - `network_agents.py` (AutoGen)
  - `service_agents.py` (LangChain)
  - `knowledge_agents.py` (LlamaIndex)
  - `customer_management_agent.py` (Custom)
- `ui/`: Streamlit UI components for Login, Admin, and Customer dashboards.
- `utils/`: Database connections and helper utilities.
- `data/`: Storage for any local data or documents.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd telecom_assistant
    ```

2.  **Install dependencies:**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory and add your necessary API keys (e.g., OpenAI API Key):
    ```env
    OPENAI_API_KEY=your_openai_api_key
    # Add other required keys as needed
    ```

## Usage

To start the application, run the Streamlit app:

```bash
streamlit run app.py
```

Navigate to the local URL provided (usually `http://localhost:8501`) to interact with the assistant.

## Technologies Used

- **Python**
- **Streamlit** (Frontend)
- **LangGraph** (Orchestration)
- **LangChain**
- **CrewAI**
- **AutoGen**
- **LlamaIndex**
- **OpenAI GPT** (LLM)
- **TextBlob** (Sentiment Analysis)
