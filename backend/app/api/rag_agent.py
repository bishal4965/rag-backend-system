from fastapi import APIRouter, HTTPException, Depends
from ..core.schemas import BookingRequest, QueryString
from sqlalchemy.orm import Session
from ..db.session import get_db
from ..agent import rag


router = APIRouter()

@router.post('/query')
async def ask_agent(request: QueryString):      # , db: Session = Depends(get_db)
    return rag.ask_agent(request.question)


@router.get('/book')
async def book_interview(request: BookingRequest, db: Session = Depends(get_db)):
    pass
