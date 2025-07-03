from typing import Optional
from langchain.tools import BaseTool
from pinecone import Pinecone
from ..core.config import settings
from ..embedding.embedder import hf
from pydantic import Field, PrivateAttr, ValidationError, BaseModel
from ..core.schemas import BookingRequest
from sqlalchemy.orm import Session

from ..db.models.booking_db import BookingInfo
from ..utils.send_mail import send_email

import re
from datetime import datetime, date


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
    _collected_data: dict = PrivateAttr()  # Track validated data between calls

    def __init__(self, db: Session):
        super().__init__()
        self._db = db
        self._collected_data = {}

    def _validate_name(self, name: str) -> str:
        """Validate full name format"""
        name = name.strip()
        if len(name) < 2:
            raise ValueError("Name must be at least 2 characters")
        if not re.match(r"^[a-zA-Z\s\-']+$", name):
            raise ValueError("Name contains invalid characters")
        return name
    
    def _validate_email(self, email: str) -> str:
        """Validate email format"""
        email = email.strip().lower()
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return email
    
    def _validate_date(self, date_str: str) -> str:
        """Validate date format and ensure it's in the future"""
        try:
            input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        
        if input_date < date.today():
            raise ValueError("Date must be in the future /  Please provide a future date (YYYY-MM-DD)." )
        return date_str
    
    def _validate_time(self, time_str: str) -> str:
        """Validate time format (supports 9am, 2:30pm, 14:30, etc.)"""
        time_str = time_str.strip().lower()
        
        # 12-hour format with AM/PM
        if re.match(r"^(0?[1-9]|1[0-2]):?([0-5][0-9])?\s*[ap]m?$", time_str):
            return time_str
        
        # 24-hour format
        if re.match(r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$", time_str):
            return time_str
        
        raise ValueError("Invalid time format. Use formats like '9am', '2:30pm', or '14:30'")
    

    def _run(self, **kwargs):
        """Handle step-by-step booking with validation"""

        for field, value in kwargs.items():
            if value is None or value == "":
                continue
                
            try:
                print(f"Validating {field}: {value}")
                if field == "full_name":
                    validated_value = self._validate_name(value)
                elif field == "email":
                    validated_value = self._validate_email(value)
                elif field == "date":
                    validated_value = self._validate_date(value)
                elif field == "time":
                    validated_value = self._validate_time(value)
                else:
                    continue  # Skip unknown fields
                
                self._collected_data[field] = validated_value
                
            except ValueError as e:
                return f"VALIDATION_ERROR: {field.upper()}_INVALID - {str(e)}"
                error_responses = {
                    "full_name": f"Invalid name: {e}. Please provide your full name again.",
                    "email": f"Invalid email: {e}. Please provide a valid email address.",
                    "date": f"Invalid date: {e}. Please provide a future date (YYYY-MM-DD).",
                    "time": f"Invalid time: {e}. Please provide a valid time (e.g., 9am or 14:30)."
                }
                return error_responses[field]


        required_fields = ["full_name", "email", "date", "time"]
        missing = [f for f in required_fields if f not in self._collected_data]
        
        if missing:
            next_field = missing[0]
            prompts = {
                "full_name": "Please provide your full name",
                "email": "What's your email address?",
                "date": "When would you like to schedule? (YYYY-MM-DD)",
                "time": "What time works for you? (e.g., 9am or 14:30)"
            }
            return prompts[next_field]
        
        try:
            booking = BookingInfo(**self._collected_data)
            self._db.add(booking)
            self._db.commit()

            send_email(
                full_name=self._collected_data["full_name"],
                receiver_email=self._collected_data["email"],
                date=self._collected_data["date"],
                time=self._collected_data["time"]
            )
            
            # Clear collected data for next booking
            self._collected_data = {}
            
            return (
                "Booking confirmed!\n\n"
                f"• Name: {booking.full_name}\n"
                f"• Email: {booking.email}\n"
                f"• Date: {booking.date}\n"
                f"• Time: {booking.time}\n\n"
                "A confirmation email has been sent!"
            )
            
        except Exception as e:
            return f"Error completing booking: {str(e)}"

        





