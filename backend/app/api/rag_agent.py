from fastapi import APIRouter, HTTPException, Depends
from ..core.schemas import BookingRequest, QueryString
from sqlalchemy.orm import Session
from ..db.session import get_db
from ..agent.rag_graph import rag_app


router = APIRouter()


@router.post('/query')
async def ask_agent(request: QueryString):      # , db: Session = Depends(get_db)
    response = rag_app.invoke({"query": request.question})
    result = response['messages'][-1].content
    return {"result": result}


@router.get('/book')
async def book_interview(request: BookingRequest, db: Session = Depends(get_db)):
    pass
