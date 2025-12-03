from langchain_community.utilities import SQLDatabase
from telecom_assistant.config.config import Config
import os

def get_database() -> SQLDatabase:
    """
    Connect to the SQLite database and return a SQLDatabase instance.
    
    Returns:
        SQLDatabase: The LangChain SQLDatabase wrapper.
        
    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    db_path = Config.DATABASE_PATH
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")
        
    # Connect to the SQLite database
    # We use the sqlite:/// prefix for SQLAlchemy
    db_uri = f"sqlite:///{db_path}"
    
    return SQLDatabase.from_uri(db_uri)

def initialize_logs_table():
    """Creates the query_logs table if it doesn't exist."""
    db = get_database()
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        customer_id TEXT,
        query_text TEXT,
        category TEXT,
        sentiment_score REAL
    );
    """
    db.run(create_table_sql)
    print("Initialized query_logs table.")

if __name__ == "__main__":
    try:
        db = get_database()
        print(f"Successfully connected to database at {Config.DATABASE_PATH}")
        print(f"Tables: {db.get_usable_table_names()}")
    except Exception as e:
        print(f"Error connecting to database: {e}")
