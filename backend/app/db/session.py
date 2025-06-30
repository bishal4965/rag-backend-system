from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from app.core.config import settings
from sqlalchemy.orm import sessionmaker


engine = create_engine(settings.SQLALCHEMY_DB_URL)

Base = declarative_base()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()