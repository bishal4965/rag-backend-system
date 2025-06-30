from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from ..db.session import get_db
from ..utils.file_utils import extract_text, chunk_text
from pinecone import Pinecone, ServerlessSpec
from ..core.config import settings
from ..embedding.embedder import create_embeddings
from ..db.models import file_db


router = APIRouter()

@router.post('/file')
async def upload_file(file: UploadFile = File(), db: Session = Depends(get_db)):
    
    # Validate file type
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file types. Allowed file types are .pdf and .txt")
    
    existing_file = db.query(file_db.FileMetadata).filter(file_db.FileMetadata.filename == file.filename).first()
    if existing_file:
        return {
            "message": "File already exists in the database. Skipping Pinecone upsert.",
            "file_id": existing_file.id
        }
    
    contents = await file.read()

    text = extract_text(contents, file.filename)
    chunks = chunk_text(text, method="recursive")
    vectors = create_embeddings(chunks)

    pc = Pinecone(settings.PINECONE_API_KEY)

    index_name = "file-embeddings-dense"
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)

    index.upsert(vectors=vectors)

    # Update database
    new_entry = file_db.FileMetadata(
        filename=file.filename,
        chunking_method=settings.CHUNKING_METHOD,
        embedding_model=settings.EMBEDDING_MODEL
        )
    
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    return {
        "message": "Embeddings are inserted and metadata is saved.",
        "file_id": new_entry.id
    }

