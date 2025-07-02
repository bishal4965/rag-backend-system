from sqlalchemy import Column, Integer, String
from ..session import Base


class BookingInfo(Base):
    __tablename__ = "booking_info"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String)
    email = Column(String)
    date = Column(String)
    time = Column(String)
