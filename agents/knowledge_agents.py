from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, SQLDatabase
from llama_index.core.query_engine import RouterQueryEngine, NLSQLTableQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool
from sqlalchemy import create_engine
from telecom_assistant.config.config import Config
import os

# Set API Key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

def create_knowledge_engine():
    """Create and return a LlamaIndex query engine for knowledge retrieval"""
    
    # Initialize LLM and Settings
    llm = OpenAI(model=Config.OPENAI_MODEL_NAME, temperature=0)
    Settings.llm = llm
    Settings.chunk_size = 1024
    
    # Load and index documents (Vector Store)
    # Use SimpleDirectoryReader to load documents from data/documents
    documents = SimpleDirectoryReader(
        input_dir=os.path.join(Config.DATA_DIR, "documents")
    ).load_data()
    
    vector_index = VectorStoreIndex.from_documents(documents)
    
    # Set up vector search query engine
    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=3
    )
    
    # Connect to the database for factual queries (SQL Store)
    db_path = Config.DATABASE_PATH
    db_uri = f"sqlite:///{db_path}"
    engine = create_engine(db_uri)
    sql_database = SQLDatabase(engine)
    
    # Create SQL query engine
    # Write prompt that helps translate natural language to SQL
    # Note: NLSQLTableQueryEngine handles the prompt internally but we can customize if needed
    # For now we rely on its default capabilities which are strong with OpenAI
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["coverage_areas", "device_compatibility", "technical_specs"]
    )
    
    # Create QueryEngineTools
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for conceptual, procedural questions like 'How do I set up VoLTE?' "
            "or 'What's the process for international roaming?'"
        ),
    )
    
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for factual, data-driven questions like 'Which areas have 5G coverage?' "
            "or 'Is the Samsung Galaxy S22 compatible with VoLTE?'"
        ),
    )
    
    # Create Router Query Engine
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            vector_tool,
            sql_tool,
        ],
    )
    
    return router_query_engine

def process_knowledge_query(query: str):
    """Process a knowledge retrieval query using the LlamaIndex query engine"""
    
    # Create or get the knowledge engine
    # In a real app, we would cache this
    engine = create_knowledge_engine()
    
    try:
        # Process the query
        response = engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error processing knowledge query: {str(e)}"

if __name__ == "__main__":
    # Test run
    print("Starting Knowledge Agent Test...")
    print(process_knowledge_query("How do I enable VoLTE?"))
