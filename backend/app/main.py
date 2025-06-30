from fastapi import FastAPI
from .api import file_upload, rag_agent

app = FastAPI()

app.include_router(file_upload.router, prefix='/upload', tags=["Upload"])
app.include_router(rag_agent.router, prefix='/agent', tags=["Agent"])