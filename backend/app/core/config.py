import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    SQLALCHEMY_DB_URL = os.getenv("DB_URL")



settings = Settings()