from fastapi import FastAPI
from .api import file_upload, rag_agent
from .db.session import Base, engine_file, engine_booking

app = FastAPI()

Base.metadata.create_all(bind=engine_file)
Base.metadata.create_all(bind=engine_booking)

app.include_router(file_upload.router, prefix='/upload', tags=["Upload"])
app.include_router(rag_agent.router, prefix='/agent', tags=["Agent"])