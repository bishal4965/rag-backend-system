from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.db.session import get_db


router = APIRouter()

@router.get('/')
def get_file():
    return 'Hi'


@router.get('/{id}')
def get_file(id: int):
    return id

@router.post('/file')
async def upload_file(file: UploadFile = File(), db: Session = Depends(get_db)):
    
    # Validate file type
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file types. Allowed file types are .pdf and .txt")
    
    contents = await file.read()

    # text = extract_text(contents, file.filename)
    return file.filename