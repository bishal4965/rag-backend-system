from sqlalchemy import Column, Integer, String
from ..session import Base


class FileMetadata(Base):
    __tablename__ = "file_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    chunking_method = Column(String)
    embedding_model = Column(String)
