from typing import Optional
from langchain.tools import BaseTool
from pinecone import Pinecone
from ..core.config import settings
from ..embedding.embedder import hf
from pydantic import Field, PrivateAttr
from ..core.schemas import BookingRequest
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..db.models.booking_db import BookingInfo
from ..utils.send_mail import send_email


class VectorSearchTool(BaseTool):
    """Tool for searching a vector db to retrieve semantically similar chunks"""

    name: str = "DocumentSearch"
    description: str =  (
        "Search for relevant documents and information based on a query. "
        "Use this tool when you need to find specific information or context "
        "to answer questions. Input should be a clear search query or question."
    )
    # query: str = Field()
    method: str = Field(default="cosine", description="The algorithm for semantic similarity search.")

    _client: Pinecone = PrivateAttr()

    def __init__(self, method: str = "cosine"):
        super().__init__()
        try:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self._client = pc.Index("file-embeddings-dense")
            self.method = method
        except Exception as e:
            print(f"Error initializing Pinecone client: {e}")
            raise

    def _run(self, query: str) -> str:
        """Execute the vector search"""
        print(f"[TOOL CALL] VectorSearchTool was invoked with query: {query}")
        query_embed = hf.embed_query(query)

        results = self._client.query(
            # namespace=settings.PINECONE_NAMESPACE
            vector=query_embed,
            top_k=3,
            include_metadata=True,
            metric=self.method,
            include_values=False
        )

        matches = getattr(results, "matches", []) 
        if not matches:
            return "No matching content found."

        chunks = [m.metadata.get("text", "") for m in matches if "text" in m.metadata]
        if not chunks:
            return "No text metadata found in the results."

        print(f"Retrieving {len(chunks)} document chunks")
        return "\n\n".join(chunks)
    

    # async def _arun(self, query: str) -> str:
    #     """Async version of the tool"""
    #     return self._run(query)


class InterviewBookingTool(BaseTool):
    """Tool for step-by-step interview booking"""
    
    name: str = "InterviewBooking"
    description: str = (
        "Book an interview by collecting details step-by-step. "
        "Call this tool multiple times as you gather information. "
        "Required fields: full_name, email, date, time."
    )

    class BookingToolArgs(BaseModel):
        full_name: Optional[str] = Field(None, description="Interviewee's full name")
        email: Optional[str] = Field(None, description="Interviewee's email address")
        date: Optional[str] = Field(None, description="Interview date (YYYY-MM-DD)")
        time: Optional[str] = Field(None, description="Interview time (e.g., 9am)")

    args_schema: type[BaseModel] = BookingToolArgs

    _db: Session = PrivateAttr()

    def __init__(self, db: Session):
        super().__init__()
        self._db = db

    
    # def _run(self, full_name: str, email: str, date: str, time: str):
    #     # info = {k: v for k, v in book_req.items() if k in ("full_name", "email", "date", "time")}
    #     # info = {
    #     #     "full_name": full_name,
    #     #     "email": email,
    #     #     "date": date,
    #     #     "time": time
    #     # }
    #     # missing = [field for field in ("full_name", "email", "date", "time") if field not in info]

    #     # if missing:
    #     #     next_field = missing[0]
    #     #     prompts = {
    #     #         "full_name": "Sure, can you provide me your name?",
    #     #         "email": "Great, what's your email address?",
    #     #         "date": "When would you like to book an appointment for your interview? (YYYY-MM-DD)",
    #     #         "time": "And at what time? (HH:MM, 24-hour)"
    #     #     }
    #     #     return prompts[next_field]

    #     # booking = BookingInfo(info)

    #     booking = BookingInfo(full_name=full_name, email=email, date=date, time=time)
    #     self._db.add(booking)
    #     self._db.commit()
    #     self._db.refresh(booking)

    #     try:
    #         send_email(
    #             full_name=booking.full_name,
    #             receiver_email=booking.email,
    #             date=booking.date,
    #             time=booking.time)
    #         return (
    #             "Appointment Summary:\n\n"
    #             f"Name: {full_name}\n"
    #             f"Email: {email}\n"
    #             f"Appointment date: {date}\n"
    #             f"Appointment time: {time}\n" 
    #         )
    #     except Exception as e:
    #         return f"Error sending an email: {str(e)}"

    def _run(self, **kwargs):
        """Handle step-by-step booking with partial info"""
        
        # Check which fields are still missing
        required = ["full_name", "email", "date", "time"]
        missing = [field for field in required if kwargs.get(field) is None]
        
        if missing:
            # Prompt for next missing field
            next_field = missing[0]
            prompts = {
                "full_name": "Sure, what's your full name?",
                "email": "Great! What's your email address?",
                "date": "When would you like to schedule? (YYYY-MM-DD)",
                "time": "What time works for you? (e.g., 9am)"
            }
            return prompts[next_field]
        
        booking = BookingInfo(**kwargs)
        self._db.add(booking)
        self._db.commit()

        try:
            send_email(
                full_name=kwargs["full_name"],
                receiver_email=kwargs["email"],
                date=kwargs["date"],
                time=kwargs["time"]
            )
            return (
                "Booking confirmed!\n\n"
                f"Name: {kwargs['full_name']}\n"
                f"Email: {kwargs['email']}\n"
                f"Date: {kwargs['date']}\n"
                f"Time: {kwargs['time']}\n\n"
                "Confirmation email sent!"
            )
        except Exception as e:
            return f"Error completing booking: {str(e)}"

        





