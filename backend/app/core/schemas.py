from pydantic import BaseModel, EmailStr


class QueryString(BaseModel):
    question: str


class BookingRequest(BaseModel):
    full_name: str
    email: EmailStr
    date: str
    time: str

    