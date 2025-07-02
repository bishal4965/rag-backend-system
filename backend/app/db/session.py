from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from ..core.config import settings
from sqlalchemy.orm import sessionmaker


engine_file = create_engine(settings.SQLALCHEMY_DB_FILE_URL)

Base = declarative_base()       # Mapper

SessionLocal_file = sessionmaker(bind=engine_file, autoflush=False, autocommit=False)

engine_booking = create_engine(settings.SQLALCHEMY_DB_BOOKING_INFO_URL)

SessionLocal_booking = sessionmaker(bind=engine_booking, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal_file()
    try:
        yield db
    finally:
        db.close()


def get_booking_db():
    db = SessionLocal_booking()
    try:
        yield db
    finally:
        db.close()