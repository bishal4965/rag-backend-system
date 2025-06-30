import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    
    SQLALCHEMY_DB_URL = os.getenv("DB_URL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    LLM_MODEL = "mistral-saba-24b"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    CHUNKING_METHOD = "recursive"






settings = Settings()