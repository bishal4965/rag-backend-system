from fastapi import APIRouter, Header
from ..core.schemas import BookingRequest, QueryString
from sqlalchemy.orm import Session
from ..db.session import get_db
from ..agent.rag_graph import rag_app
from langchain_core.messages import AIMessage


router = APIRouter()


@router.post('/query')
async def ask_agent(request: QueryString, thread_id: str = Header(..., description="Unique conversation ID")): 
    
    response = rag_app.invoke(
        {"query": request.question},
        config={"thread_id": thread_id}
        )
    # result = response['messages'][-1].content
    for msg in reversed(response["messages"]):
        if isinstance(msg, AIMessage):
            return {"result": msg.content}
    return {"result": "No response from the agent"}


# @router.get('/book')
# async def book_interview(request: BookingRequest, db: Session = Depends(get_db)):
#     pass
