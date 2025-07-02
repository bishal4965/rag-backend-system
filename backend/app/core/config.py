import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    
    SQLALCHEMY_DB_FILE_URL = os.getenv("DB_URL")
    SQLALCHEMY_DB_BOOKING_INFO_URL = os.getenv("BOOKING_INFO_DB_URL")

    REDIS_CACHE_URL = "redis://localhost:6379/0"       # /0 refer to db index: db numbered 0
    REDIS_MEMORY_URL = "redis://localhost:6379/0"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    MAILTRAP_API_KEY = os.getenv("MAILTRAP_API_KEY")

    LLM_MODEL = "mistral-saba-24b"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    CHUNKING_METHOD = "recursive"

    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    SENDER_MAIL_PASSWORD = os.getenv("SENDER_EMAIL_PASSWORD")
    






settings = Settings()